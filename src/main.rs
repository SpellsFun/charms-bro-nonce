use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{RwLock, Semaphore};

use bro::{run_search, SearchConfig, SearchOutcome, MAX_ILP};

#[derive(Clone)]
struct AppState {
    jobs: Arc<RwLock<HashMap<String, Job>>>,
    concurrency: Arc<Semaphore>,
    auth_token: Option<String>,
    log_file: Option<Arc<Mutex<File>>>,
    poll_tracker: Arc<RwLock<HashMap<String, PollInfo>>>,
}

#[derive(Clone)]
struct PollInfo {
    last_logged: SystemTime,
    count: u32,
    last_summary: SystemTime,
}

struct Job {
    status: JobStatus,
    submitted_at: SystemTime,
}

enum JobStatus {
    Pending,
    Running {
        started_at: SystemTime,
    },
    Completed {
        started_at: SystemTime,
        finished_at: SystemTime,
        outcome: SearchOutcome,
    },
    Failed {
        started_at: Option<SystemTime>,
        finished_at: SystemTime,
        message: String,
    },
}

#[derive(Deserialize)]
struct CreateJobRequest {
    outpoint: String,
    #[serde(default)]
    options: Option<SearchOptions>,
    #[serde(default)]
    wait: Option<bool>,
}

#[derive(Deserialize, Default, Debug)]
struct SearchOptions {
    total_nonce: Option<u64>,
    start_nonce: Option<u64>,
    batch_size: Option<u64>,
    threads_per_block: Option<u32>,
    blocks: Option<u32>,
    binary_nonce: Option<bool>,
    persistent: Option<bool>,
    chunk_size: Option<u32>,
    ilp: Option<u32>,
    progress_ms: Option<u64>,
    odometer: Option<bool>,
    gpu_ids: Option<Vec<u32>>,
    gpu_weights: Option<Vec<f64>>,
}

#[derive(Serialize)]
struct CreateJobResponse {
    job_id: String,
    status: String,
    result: Option<SearchOutcome>,
    error: Option<String>,
}

#[derive(Serialize)]
struct JobResponse {
    job_id: String,
    status: String,
    submitted_at: u64,
    started_at: Option<u64>,
    finished_at: Option<u64>,
    result: Option<SearchOutcome>,
    error: Option<String>,
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({ "error": self.message }));
        (self.status, body).into_response()
    }
}

async fn submit_job(
    state: &AppState,
    job_id: String,
    config: SearchConfig,
    wait: bool,
) -> Result<(StatusCode, CreateJobResponse), ApiError> {
    let permit = state
        .concurrency
        .clone()
        .acquire_owned()
        .await
        .map_err(|e| ApiError::internal(format!("failed to acquire worker permit: {e}")))?;

    let submitted_at = SystemTime::now();
    {
        let mut jobs = state.jobs.write().await;
        // 再次检查，避免并发问题
        if jobs.contains_key(&job_id) {
            return Err(ApiError::bad_request(format!("Job {} already exists", job_id)));
        }
        jobs.insert(
            job_id.clone(),
            Job {
                status: JobStatus::Pending,
                submitted_at,
            },
        );
    }

    // 记录任务开始
    log_print(
        state.log_file.as_ref(),
        &format!("[Job Started] Outpoint: {}", job_id),
    );

    let runner_state = state.clone();
    if wait {
        let _permit = permit;
        let started_at = SystemTime::now();
        {
            let mut jobs = runner_state.jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.status = JobStatus::Running { started_at };
            }
        }

        let result = tokio::task::spawn_blocking(move || run_search(config)).await;
        let finished_at = SystemTime::now();

        let mut status = "completed".to_string();
        let mut outcome_opt: Option<SearchOutcome> = None;
        let mut error_opt: Option<String> = None;
        let mut http_status = StatusCode::OK;

        match result {
            Ok(Ok(outcome)) => {
                outcome_opt = Some(outcome.clone());
                log_print(
                    runner_state.log_file.as_ref(),
                    &format!(
                        "[Job Completed] Outpoint: {}, best_lz: {}, best_nonce: {}",
                        job_id, outcome.best_lz, outcome.best_nonce
                    ),
                );
                let mut jobs = runner_state.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Completed {
                        started_at,
                        finished_at,
                        outcome,
                    };
                }
            }
            Ok(Err(err)) => {
                status = "failed".to_string();
                error_opt = Some(err.to_string());
                http_status = StatusCode::INTERNAL_SERVER_ERROR;
                log_print(
                    runner_state.log_file.as_ref(),
                    &format!("[Job Failed] Outpoint: {}, Error: {}", job_id, err),
                );
                let mut jobs = runner_state.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed {
                        started_at: Some(started_at),
                        finished_at,
                        message: err.to_string(),
                    };
                }
            }
            Err(join_err) => {
                status = "failed".to_string();
                let msg = format!("worker panic: {join_err}");
                error_opt = Some(msg.clone());
                http_status = StatusCode::INTERNAL_SERVER_ERROR;
                let mut jobs = runner_state.jobs.write().await;
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.status = JobStatus::Failed {
                        started_at: Some(started_at),
                        finished_at,
                        message: msg,
                    };
                }
            }
        }

        return Ok((
            http_status,
            CreateJobResponse {
                job_id,
                status,
                result: outcome_opt,
                error: error_opt,
            },
        ));
    }

    let job_id_for_task = job_id.clone();
    tokio::spawn(async move {
        let _permit = permit;
        let started_at = SystemTime::now();
        {
            let mut jobs = runner_state.jobs.write().await;
            if let Some(job) = jobs.get_mut(&job_id_for_task) {
                job.status = JobStatus::Running { started_at };
            }
        }

        let result = tokio::task::spawn_blocking(move || run_search(config)).await;
        let finished_at = SystemTime::now();
        let mut jobs = runner_state.jobs.write().await;
        if let Some(job) = jobs.get_mut(&job_id_for_task) {
            match result {
                Ok(Ok(outcome)) => {
                    log_print(
                        runner_state.log_file.as_ref(),
                        &format!(
                            "[Job Completed] Outpoint: {}, best_lz: {}, best_nonce: {}",
                            job_id_for_task, outcome.best_lz, outcome.best_nonce
                        ),
                    );
                    job.status = JobStatus::Completed {
                        started_at,
                        finished_at,
                        outcome,
                    };
                }
                Ok(Err(err)) => {
                    log_print(
                        runner_state.log_file.as_ref(),
                        &format!("[Job Failed] Outpoint: {}, Error: {}", job_id_for_task, err),
                    );
                    job.status = JobStatus::Failed {
                        started_at: Some(started_at),
                        finished_at,
                        message: err.to_string(),
                    };
                }
                Err(join_err) => {
                    job.status = JobStatus::Failed {
                        started_at: Some(started_at),
                        finished_at,
                        message: format!("worker panic: {join_err}"),
                    };
                }
            }
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        CreateJobResponse {
            job_id,
            status: "pending".to_string(),
            result: None,
            error: None,
        },
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8001u16);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    // 初始化日志文件（默认使用./bro-api.log）
    let log_path = std::env::var("LOG_FILE").unwrap_or_else(|_| "./bro-api.log".to_string());
    let log_file = match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(file) => {
            println!("Logging to file: {}", log_path);
            Some(Arc::new(Mutex::new(file)))
        }
        Err(e) => {
            eprintln!("Failed to open log file {}: {}", log_path, e);
            None
        }
    };

    // 启动日志
    log_print(
        log_file.as_ref(),
        &format!("========== Starting bro API server =========="),
    );
    log_print(
        log_file.as_ref(),
        &format!("Time: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")),
    );
    log_print(
        log_file.as_ref(),
        "Note: Job cache cleared on restart. Completed jobs may run again if resubmitted.",
    );

    // 从环境变量获取认证token
    let auth_token = std::env::var("AUTH_TOKEN").ok();
    if let Some(ref token) = auth_token {
        log_print(
            log_file.as_ref(),
            &format!("API authentication enabled with token: {}...", &token[..token.len().min(8)]),
        );
    } else {
        log_print(
            log_file.as_ref(),
            "WARNING: API authentication disabled (no AUTH_TOKEN set)",
        );
    }

    let state = AppState {
        jobs: Arc::new(RwLock::new(HashMap::new())),
        concurrency: Arc::new(Semaphore::new(1)),
        auth_token,
        log_file: log_file.clone(),
        poll_tracker: Arc::new(RwLock::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/api/v1/jobs", post(create_job).get(list_jobs))
        .route("/api/v1/jobs/{id}", get(get_job))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state.clone());

    log_print(
        log_file.as_ref(),
        &format!("Server listening on {}", addr),
    );

    // 启动清理任务，定期清理过期的记录
    let cleaner_state = state.clone();
    let log_file_for_cleaner = log_file.clone();
    tokio::spawn(async move {
        // 可配置的保留时间（通过环境变量）
        let keep_completed_hours = std::env::var("KEEP_COMPLETED_HOURS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(24); // 默认24小时
        let keep_failed_hours = std::env::var("KEEP_FAILED_HOURS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1); // 默认1小时

        log_print(
            log_file_for_cleaner.as_ref(),
            &format!("[Cleanup] Task retention: completed={}h, failed={}h",
                    keep_completed_hours, keep_failed_hours),
        );

        let mut interval = tokio::time::interval(Duration::from_secs(300)); // 每5分钟执行一次
        loop {
            interval.tick().await;
            let now = SystemTime::now();

            // 清理轮询记录
            {
                let mut tracker = cleaner_state.poll_tracker.write().await;
                let before = tracker.len();
                tracker.retain(|_, info| {
                    // 保留最近5分钟内有活动的记录
                    now.duration_since(info.last_logged)
                        .unwrap_or(Duration::from_secs(0))
                        < Duration::from_secs(300)
                });
                if before > tracker.len() {
                    log_print(
                        cleaner_state.log_file.as_ref(),
                        &format!("[Cleanup] Removed {} inactive poll records", before - tracker.len()),
                    );
                }
            }

            // 清理任务记录
            {
                let mut jobs = cleaner_state.jobs.write().await;
                let before = jobs.len();
                jobs.retain(|id, job| {
                    match &job.status {
                        // 保留正在运行的任务
                        JobStatus::Running { .. } => true,
                        // 保留最近的待处理任务（1小时内）
                        JobStatus::Pending => {
                            now.duration_since(job.submitted_at)
                                .unwrap_or(Duration::from_secs(0))
                                < Duration::from_secs(3600)
                        }
                        // 保留最近完成的任务
                        JobStatus::Completed { finished_at, .. } => {
                            now.duration_since(*finished_at)
                                .unwrap_or(Duration::from_secs(0))
                                < Duration::from_secs(keep_completed_hours * 3600)
                        }
                        // 保留最近失败的任务（允许重试）
                        JobStatus::Failed { finished_at, .. } => {
                            now.duration_since(*finished_at)
                                .unwrap_or(Duration::from_secs(0))
                                < Duration::from_secs(keep_failed_hours * 3600)
                        }
                    }
                });
                if before > jobs.len() {
                    log_print(
                        cleaner_state.log_file.as_ref(),
                        &format!("[Cleanup] Removed {} expired jobs (completed: 24h, failed/pending: 1h)",
                                before - jobs.len()),
                    );
                }
            }
        }
    });

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

async fn create_job(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<AppState>,
    Json(payload): Json<CreateJobRequest>,
) -> Result<(StatusCode, Json<CreateJobResponse>), ApiError> {
    let outpoint = payload.outpoint.trim().to_string();

    // 记录请求日志
    log_print(
        state.log_file.as_ref(),
        &format!(
            "[Request] IP: {}, Outpoint: {}, Wait: {}",
            addr.ip(),
            outpoint,
            payload.wait.unwrap_or(false)
        ),
    );

    if outpoint.is_empty() {
        return Err(ApiError::bad_request("outpoint must not be empty"));
    }

    let mut config = SearchConfig::with_outpoint(outpoint.clone());
    if let Some(opts) = payload.options {
        apply_options(&mut config, opts)?;
    }
    let wait = payload.wait.unwrap_or(false);

    // 检查是否已有相同outpoint的任务
    {
        let jobs = state.jobs.read().await;
        if let Some(existing_job) = jobs.get(&outpoint) {
            // 如果任务正在运行或已完成，直接返回现有任务信息
            match &existing_job.status {
                JobStatus::Running { .. } => {
                    log_print(
                        state.log_file.as_ref(),
                        &format!(
                            "[Duplicate] IP: {} tried to submit outpoint: {} (already running)",
                            addr.ip(),
                            outpoint
                        ),
                    );
                    return Ok((StatusCode::OK, Json(existing_job.to_create_response(&outpoint))));
                }
                JobStatus::Completed { .. } => {
                    log_print(
                        state.log_file.as_ref(),
                        &format!(
                            "[Duplicate] IP: {} tried to submit outpoint: {} (already completed)",
                            addr.ip(),
                            outpoint
                        ),
                    );
                    // 返回已完成的结果，永远不删除已完成的任务
                    return Ok((StatusCode::OK, Json(existing_job.to_create_response(&outpoint))));
                }
                JobStatus::Pending => {
                    // 如果是Pending状态，可能是之前失败的任务，继续检查
                    log_print(
                        state.log_file.as_ref(),
                        &format!(
                            "[Info] Reusing pending job for outpoint: {}",
                            outpoint
                        ),
                    );
                    return Ok((StatusCode::OK, Json(existing_job.to_create_response(&outpoint))));
                }
                JobStatus::Failed { .. } => {
                    // 如果之前失败了，允许重试
                    log_print(
                        state.log_file.as_ref(),
                        &format!(
                            "[Retry] Retrying failed job for outpoint: {}",
                            outpoint
                        ),
                    );
                    // 继续执行，下面会删除失败的记录
                }
            }
        }
    }

    // 只删除失败的任务记录以允许重试
    {
        let mut jobs = state.jobs.write().await;
        if let Some(job) = jobs.get(&outpoint) {
            if matches!(job.status, JobStatus::Failed { .. }) {
                jobs.remove(&outpoint);
            }
        }
    }

    let (status, response) = submit_job(&state, outpoint, config, wait).await?;
    Ok((status, Json(response)))
}

async fn get_job(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    // 智能轮询日志记录
    let poll_key = format!("{}:{}", addr.ip(), id);
    let now = SystemTime::now();
    let should_log = {
        let mut tracker = state.poll_tracker.write().await;
        let info = tracker.entry(poll_key.clone()).or_insert(PollInfo {
            last_logged: now,
            count: 0,
            last_summary: now,
        });

        info.count += 1;

        // 每30秒记录一次详细日志
        let time_since_log = now.duration_since(info.last_logged).unwrap_or(Duration::from_secs(0));
        let time_since_summary = now.duration_since(info.last_summary).unwrap_or(Duration::from_secs(0));

        // 每分钟输出汇总
        if time_since_summary >= Duration::from_secs(60) && info.count > 1 {
            log_print(
                state.log_file.as_ref(),
                &format!(
                    "[Poll Summary] IP: {}, Job: {} - {} polls in last minute",
                    addr.ip(), id, info.count
                ),
            );
            info.count = 0;
            info.last_summary = now;
            info.last_logged = now;
            false  // 汇总后不记录单次请求
        } else if time_since_log >= Duration::from_secs(30) || info.count == 1 {
            // 首次或超过30秒记录
            info.last_logged = now;
            true
        } else {
            false
        }
    };

    if should_log {
        log_print(
            state.log_file.as_ref(),
            &format!("[Request] IP: {}, Get Job: {}", addr.ip(), id),
        );
    }
    // 只查询，不创建新任务
    let jobs = state.jobs.read().await;
    if let Some(job) = jobs.get(&id) {
        Ok((StatusCode::OK, Json(job.to_response(&id))))
    } else {
        // 任务不存在，返回404
        Err(ApiError {
            status: StatusCode::NOT_FOUND,
            message: format!("Job not found: {}", id),
        })
    }
}

async fn list_jobs(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<AppState>,
) -> Result<Json<Vec<JobResponse>>, ApiError> {
    // 对list操作不进行去重，因为它们通常不会频繁轮询
    log_print(
        state.log_file.as_ref(),
        &format!("[Request] IP: {}, List Jobs", addr.ip()),
    );
    let jobs = state.jobs.read().await;
    let mut entries: Vec<JobResponse> = jobs.iter().map(|(id, job)| job.to_response(id)).collect();
    entries.sort_by_key(|resp| resp.submitted_at);
    Ok(Json(entries))
}

impl Job {
    fn to_create_response(&self, id: &str) -> CreateJobResponse {
        let (status, result, error) = match &self.status {
            JobStatus::Pending => ("pending".to_string(), None, None),
            JobStatus::Running { .. } => ("running".to_string(), None, None),
            JobStatus::Completed { outcome, .. } => {
                ("completed".to_string(), Some(outcome.clone()), None)
            }
            JobStatus::Failed { message, .. } => {
                ("failed".to_string(), None, Some(message.clone()))
            }
        };

        CreateJobResponse {
            job_id: id.to_string(),
            status,
            result,
            error,
        }
    }

    fn to_response(&self, id: &str) -> JobResponse {
        let (status, started_at, finished_at, result, error) = match &self.status {
            JobStatus::Pending => ("pending".to_string(), None, None, None, None),
            JobStatus::Running { started_at } => (
                "running".to_string(),
                Some(system_time_to_secs(*started_at)),
                None,
                None,
                None,
            ),
            JobStatus::Completed {
                started_at,
                finished_at,
                outcome,
            } => (
                "completed".to_string(),
                Some(system_time_to_secs(*started_at)),
                Some(system_time_to_secs(*finished_at)),
                Some(outcome.clone()),
                None,
            ),
            JobStatus::Failed {
                started_at,
                finished_at,
                message,
            } => (
                "failed".to_string(),
                started_at.map(|t| system_time_to_secs(t)),
                Some(system_time_to_secs(*finished_at)),
                None,
                Some(message.clone()),
            ),
        };

        JobResponse {
            job_id: id.to_string(),
            status,
            submitted_at: system_time_to_secs(self.submitted_at),
            started_at,
            finished_at,
            result,
            error,
        }
    }
}

fn system_time_to_secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs()
}

// 认证中间件
async fn auth_middleware(
    State(state): State<AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 如果设置了认证token，则验证
    if let Some(ref expected_token) = state.auth_token {
        let auth_header = headers
            .get("authorization")
            .and_then(|h| h.to_str().ok());

        let valid = match auth_header {
            Some(header) if header.starts_with("Bearer ") => {
                let token = &header[7..];
                token == expected_token
            }
            _ => false,
        };

        if !valid {
            // 尝试从请求中提取outpoint信息
            let path = request.uri().path().to_string();
            let method = request.method().clone();

            let outpoint_info = if path.starts_with("/api/v1/jobs") {
                if method == Method::POST {
                    // 对于POST请求，读取body中的outpoint（因为认证失败，后续不会使用body）
                    let (_parts, body) = request.into_parts();

                    // 使用axum的方法读取body
                    match axum::body::to_bytes(body, usize::MAX).await {
                        Ok(bytes) => {
                            if let Ok(json_str) = std::str::from_utf8(&bytes) {
                                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_str) {
                                    if let Some(outpoint) = json_value.get("outpoint").and_then(|v| v.as_str()) {
                                        format!("Outpoint: {}", outpoint)
                                    } else {
                                        "POST request (no outpoint in body)".to_string()
                                    }
                                } else {
                                    "POST request (invalid JSON)".to_string()
                                }
                            } else {
                                "POST request (invalid UTF-8)".to_string()
                            }
                        }
                        Err(_) => "POST request (failed to read body)".to_string(),
                    }
                } else if path.starts_with("/api/v1/jobs/") {
                    // 对于GET /api/v1/jobs/{id}，id就是outpoint
                    let id = path.trim_start_matches("/api/v1/jobs/");
                    format!("Outpoint: {}", id)
                } else {
                    "List request".to_string()
                }
            } else {
                format!("Unknown path: {}", path)
            };

            log_print(
                state.log_file.as_ref(),
                &format!(
                    "[Auth Failed] IP: {}, Path: {}, {}",
                    addr.ip(),
                    path,
                    outpoint_info
                ),
            );
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    Ok(next.run(request).await)
}

// 日志输出函数，同时输出到控制台和文件
fn log_print(log_file: Option<&Arc<Mutex<File>>>, message: &str) {
    println!("{}", message);

    if let Some(file) = log_file {
        if let Ok(mut f) = file.lock() {
            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let _ = writeln!(f, "[{}] {}", timestamp, message);
            let _ = f.flush();
        }
    }
}

fn apply_options(config: &mut SearchConfig, opts: SearchOptions) -> Result<(), ApiError> {
    // 记录接收到的参数
    let mut applied = Vec::new();

    if let Some(v) = opts.total_nonce {
        // 限制total_nonce最大为2万亿
        let limited_v = if v > bro::MAX_TOTAL_NONCE {
            println!("[{}] [Warning] total_nonce {} exceeds max limit, capping to {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                v,
                bro::MAX_TOTAL_NONCE
            );
            bro::MAX_TOTAL_NONCE
        } else {
            v
        };
        config.total_nonce_all = limited_v;
        applied.push(format!("total_nonce={}", limited_v));
    }
    if let Some(v) = opts.start_nonce {
        config.start_nonce_all = v;
        applied.push(format!("start_nonce={}", v));
    }
    if let Some(v) = opts.batch_size {
        config.batch_size = v;
        applied.push(format!("batch_size={}", v));
    }
    if let Some(v) = opts.threads_per_block {
        config.threads_per_block = v;
        applied.push(format!("threads={}", v));
    }
    if let Some(v) = opts.blocks {
        config.blocks = v;
        applied.push(format!("blocks={}", v));
    }
    if let Some(v) = opts.binary_nonce {
        config.binary_nonce = v;
        applied.push(format!("binary_nonce={}", v));
    }
    if let Some(v) = opts.persistent {
        config.persistent = v;
        applied.push(format!("persistent={}", v));
    }
    if let Some(v) = opts.chunk_size {
        config.chunk_size = v;
        applied.push(format!("chunk_size={}", v));
    }
    if let Some(v) = opts.ilp {
        config.ilp = v.clamp(1, MAX_ILP);
        applied.push(format!("ilp={}", config.ilp));
    }
    if let Some(v) = opts.progress_ms {
        config.progress_ms = v;
        applied.push(format!("progress_ms={}", v));
    }
    if let Some(v) = opts.odometer {
        config.odometer = v;
        applied.push(format!("odometer={}", v));
    }
    if let Some(ids) = opts.gpu_ids {
        config.gpu_ids = Some(ids.clone());
        applied.push(format!("gpu_ids={:?}", ids));
    }
    if let Some(ws) = opts.gpu_weights {
        config.gpu_weights = Some(ws.clone());
        applied.push(format!("gpu_weights={:?}", ws));
    }

    // 输出所有接收到的参数
    if !applied.is_empty() {
        // 注意：这里没有state可用，所以只能输出到控制台
        println!("[{}] [Config] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            applied.join(", "));
    }

    Ok(())
}
