use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{RwLock, Semaphore};

use bro::{run_search, SearchConfig, SearchOutcome};

#[derive(Clone)]
struct AppState {
    jobs: Arc<RwLock<HashMap<String, Job>>>,
    concurrency: Arc<Semaphore>,
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

#[derive(Deserialize, Default)]
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
    min_best_lz: Option<u32>,
    min_total_nonce: Option<u64>,
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
        jobs.insert(
            job_id.clone(),
            Job {
                status: JobStatus::Pending,
                submitted_at,
            },
        );
    }

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
                    job.status = JobStatus::Completed {
                        started_at,
                        finished_at,
                        outcome,
                    };
                }
                Ok(Err(err)) => {
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
        .unwrap_or(3000u16);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    let state = AppState {
        jobs: Arc::new(RwLock::new(HashMap::new())),
        concurrency: Arc::new(Semaphore::new(1)),
    };

    let app = Router::new()
        .route("/api/v1/jobs", post(create_job).get(list_jobs))
        .route("/api/v1/jobs/{id}", get(get_job))
        .with_state(state.clone());

    println!("bro API server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn create_job(
    State(state): State<AppState>,
    Json(payload): Json<CreateJobRequest>,
) -> Result<(StatusCode, Json<CreateJobResponse>), ApiError> {
    let outpoint = payload.outpoint.trim().to_string();
    if outpoint.is_empty() {
        return Err(ApiError::bad_request("outpoint must not be empty"));
    }

    let mut config = SearchConfig::with_outpoint(outpoint.clone());
    if let Some(opts) = payload.options {
        apply_options(&mut config, opts)?;
    }
    let wait = payload.wait.unwrap_or(false);
    let (status, response) = submit_job(&state, outpoint, config, wait).await?;
    Ok((status, Json(response)))
}

async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<JobResponse>), ApiError> {
    if let Some(resp) = {
        let jobs = state.jobs.read().await;
        jobs.get(&id).map(|job| job.to_response(&id))
    } {
        return Ok((StatusCode::OK, Json(resp)));
    }

    let (status, _) = submit_job(
        &state,
        id.clone(),
        SearchConfig::with_outpoint(id.clone()),
        false,
    )
    .await?;
    let jobs = state.jobs.read().await;
    let job = jobs
        .get(&id)
        .ok_or_else(|| ApiError::internal("job missing after submission"))?;
    Ok((status, Json(job.to_response(&id))))
}

async fn list_jobs(State(state): State<AppState>) -> Result<Json<Vec<JobResponse>>, ApiError> {
    let jobs = state.jobs.read().await;
    let mut entries: Vec<JobResponse> = jobs.iter().map(|(id, job)| job.to_response(id)).collect();
    entries.sort_by_key(|resp| resp.submitted_at);
    Ok(Json(entries))
}

impl Job {
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

fn apply_options(config: &mut SearchConfig, opts: SearchOptions) -> Result<(), ApiError> {
    if let Some(v) = opts.total_nonce {
        config.total_nonce_all = v;
    }
    if let Some(v) = opts.start_nonce {
        config.start_nonce_all = v;
    }
    if let Some(v) = opts.batch_size {
        config.batch_size = v;
    }
    if let Some(v) = opts.threads_per_block {
        config.threads_per_block = v;
    }
    if let Some(v) = opts.blocks {
        config.blocks = v;
    }
    if let Some(v) = opts.binary_nonce {
        config.binary_nonce = v;
    }
    if let Some(v) = opts.persistent {
        config.persistent = v;
    }
    if let Some(v) = opts.chunk_size {
        config.chunk_size = v;
    }
    if let Some(v) = opts.ilp {
        config.ilp = v;
    }
    if let Some(v) = opts.progress_ms {
        config.progress_ms = v;
    }
    if let Some(v) = opts.odometer {
        config.odometer = v;
    }
    if let Some(ids) = opts.gpu_ids {
        config.gpu_ids = Some(ids);
    }
    if let Some(ws) = opts.gpu_weights {
        config.gpu_weights = Some(ws);
    }
    if let Some(v) = opts.min_best_lz {
        config.min_best_lz = Some(v);
    }
    if let Some(v) = opts.min_total_nonce {
        config.min_total_nonce = Some(v);
    }

    Ok(())
}
