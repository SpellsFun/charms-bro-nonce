use anyhow::{anyhow, Result};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use hex::encode as hex_encode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, env, net::SocketAddr, sync::Arc, time::Instant};
use tokio::sync::Mutex;

#[derive(Clone)]
struct AppState {
    miner: Arc<gpu::Miner>,
    jobs: JobStore,
}

type JobStore = Arc<Mutex<HashMap<String, JobEntry>>>;

#[derive(Clone)]
struct JobEntry {
    persistent: bool,
    state: JobState,
    result: Option<JobResult>,
    error: Option<String>,
}

#[derive(Clone)]
struct JobParameters {
    outpoint: String,
    start_nonce: u64,
    total_nonce: u64,
    batch_size: u32,
    threads_per_block: u32,
    blocks: Option<u32>,
    ilp: u32,
}

#[derive(Serialize, Clone)]
struct JobResult {
    outpoint: String,
    best_nonce: String,
    best_lz: u32,
    best_hash: String,
    searched: u64,
    start_nonce: String,
    duration_ms: u64,
    throughput_ghs: f64,
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum JobState {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Serialize)]
struct JobStatusResponse {
    outpoint: String,
    persistent: bool,
    status: JobState,
    result: Option<JobResult>,
    error: Option<String>,
}

impl From<(String, &JobEntry)> for JobStatusResponse {
    fn from(value: (String, &JobEntry)) -> Self {
        let (outpoint, entry) = value;
        Self {
            outpoint,
            persistent: entry.persistent,
            status: entry.state,
            result: entry.result.clone(),
            error: entry.error.clone(),
        }
    }
}

#[derive(Deserialize)]
struct CreateJobRequest {
    outpoint: String,
    start_nonce: Option<u64>,
    /// 总共要尝试的 nonce 数（可以非常大）。未提供时默认 5000 亿。
    total_nonce: Option<u64>,
    /// 单次 GPU 调度的 batch 大小（默认 1,000,000），用于拆分大任务。
    batch_size: Option<u32>,
    /// 每个 block 的线程数（默认 256）。
    threads_per_block: Option<u32>,
    /// kernel grid 的 block 数，未提供时自动根据 batch_size 推导。
    blocks: Option<u32>,
    /// 每个线程处理的 nonce 数（ILP），默认 1。
    ilp: Option<u32>,
    /// 是否启用持久化：若为 true，完成后相同 outpoint 的请求会直接返回缓存结果。
    persistent: Option<bool>,
    /// 若为 true，等待任务执行完成后再返回，便于测试/同步调用。
    wait: Option<bool>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let miner = Arc::new(gpu::Miner::new()?);
    let jobs: JobStore = Arc::new(Mutex::new(HashMap::new()));
    let state = AppState { miner, jobs };

    let app = Router::new()
        .route("/api/v1/jobs", post(create_job))
        .route("/api/v1/jobs/{outpoint}", get(get_job))
        .with_state(state);

    let port = env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8001);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    println!("listening on http://{local_addr}");
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    let outpoint = req.outpoint.trim();
    if outpoint.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    if !outpoint.contains(':') {
        return Err(StatusCode::BAD_REQUEST);
    }

    let persistent = req.persistent.unwrap_or(false);
    let wait_for_completion = req.wait.unwrap_or(false);

    let params = JobParameters {
        outpoint: outpoint.to_string(),
        start_nonce: req.start_nonce.unwrap_or(0),
        total_nonce: req.total_nonce.unwrap_or(500_000_000_000),
        batch_size: req.batch_size.unwrap_or(1_000_000).max(1),
        threads_per_block: req.threads_per_block.unwrap_or(256).max(1),
        blocks: req.blocks.map(|b| b.max(1)),
        ilp: req.ilp.unwrap_or(1).max(1),
    };

    if params.total_nonce == 0 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let mut jobs = state.jobs.lock().await;
    if let Some(existing) = jobs.get(outpoint) {
        match existing.state {
            JobState::Completed if persistent => {
                let response = JobStatusResponse::from((outpoint.to_string(), existing));
                return Ok((StatusCode::OK, Json(response)));
            }
            JobState::Pending | JobState::Running => {
                let response = JobStatusResponse::from((outpoint.to_string(), existing));
                return Ok((StatusCode::ACCEPTED, Json(response)));
            }
            JobState::Completed | JobState::Failed => {
                if existing.persistent && persistent {
                    let response = JobStatusResponse::from((outpoint.to_string(), existing));
                    return Ok((StatusCode::OK, Json(response)));
                }
                // fallthrough to recreate job
            }
        }
    }

    let entry = JobEntry {
        persistent,
        state: JobState::Pending,
        result: None,
        error: None,
    };
    jobs.insert(outpoint.to_string(), entry);
    drop(jobs);

    let jobs_handle = state.jobs.clone();
    let miner = state.miner.clone();
    let job_id = outpoint.to_string();
    let params_for_task = params.clone();

    if wait_for_completion {
        {
            let mut jobs = jobs_handle.lock().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = JobState::Running;
            }
        }

        let outcome = tokio::task::spawn_blocking(move || run_job(&miner, params_for_task)).await;

        let mut jobs = jobs_handle.lock().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            match outcome {
                Ok(Ok(result)) => {
                    job.state = JobState::Completed;
                    job.result = Some(result);
                    job.error = None;
                }
                Ok(Err(err)) => {
                    job.state = JobState::Failed;
                    job.error = Some(err.to_string());
                }
                Err(join_err) => {
                    job.state = JobState::Failed;
                    job.error = Some(join_err.to_string());
                }
            }
        }

        let response = jobs
            .get(&job_id)
            .map(|entry| JobStatusResponse::from((job_id, entry)))
            .unwrap();

        let status = if response.status == JobState::Completed {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };

        return Ok((status, Json(response)));
    }

    tokio::spawn(async move {
        {
            let mut jobs = jobs_handle.lock().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = JobState::Running;
            }
        }

        let mining = tokio::task::spawn_blocking(move || run_job(&miner, params_for_task));
        let outcome = mining.await;

        let mut jobs = jobs_handle.lock().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            match outcome {
                Ok(Ok(result)) => {
                    job.state = JobState::Completed;
                    job.result = Some(result);
                    job.error = None;
                }
                Ok(Err(err)) => {
                    job.state = JobState::Failed;
                    job.error = Some(err.to_string());
                }
                Err(join_err) => {
                    job.state = JobState::Failed;
                    job.error = Some(join_err.to_string());
                }
            }
        }
    });

    let jobs = state.jobs.lock().await;
    let response = jobs
        .get(outpoint)
        .map(|entry| JobStatusResponse::from((outpoint.to_string(), entry)))
        .unwrap();

    Ok((StatusCode::ACCEPTED, Json(response)))
}

async fn get_job(
    State(state): State<AppState>,
    Path(outpoint): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    let jobs = state.jobs.lock().await;
    if let Some(entry) = jobs.get(&outpoint) {
        let response = JobStatusResponse::from((outpoint, entry));
        Ok((StatusCode::OK, Json(response)))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

fn run_job(miner: &gpu::Miner, params: JobParameters) -> Result<JobResult> {
    let started = Instant::now();
    let mut searched = 0u64;
    let mut best_lz = None;
    let mut best_nonce = 0u64;
    let mut best_hash = String::new();

    let batch_limit = u64::from(params.batch_size);
    let threads_per_block_u64 = u64::from(params.threads_per_block);
    let ilp_u64 = u64::from(params.ilp);
    let explicit_capacity = params
        .blocks
        .map(|b| u64::from(b) * threads_per_block_u64 * ilp_u64);

    while searched < params.total_nonce {
        let remaining = params.total_nonce - searched;
        let launch_capacity = explicit_capacity.unwrap_or(batch_limit);
        let desired = remaining.min(launch_capacity).min(u64::from(u32::MAX));
        if desired == 0 {
            break;
        }

        let blocks = params.blocks.unwrap_or_else(|| {
            let threads_needed = ((desired + ilp_u64 - 1) / ilp_u64).max(1);
            let blocks_needed =
                ((threads_needed + threads_per_block_u64 - 1) / threads_per_block_u64).max(1);
            blocks_needed.min(u64::from(u32::MAX)) as u32
        });

        let current_batch = desired as u32;
        let batch_start = params
            .start_nonce
            .checked_add(searched)
            .ok_or_else(|| anyhow!("start_nonce overflow"))?;

        let (nonce, lz, hash) = miner.mine_batch(
            params.outpoint.as_bytes(),
            batch_start,
            current_batch,
            blocks,
            params.threads_per_block,
            params.ilp,
        )?;

        match best_lz {
            Some(current_best) if lz < current_best => {}
            Some(current_best) if lz == current_best && nonce >= best_nonce => {}
            _ => {
                best_lz = Some(lz);
                best_nonce = nonce;
                best_hash = hash;
            }
        }

        searched += u64::from(current_batch);
    }

    let best_lz_gpu = best_lz.unwrap_or(0);
    let elapsed = started.elapsed();
    let duration_ms = elapsed.as_millis() as u64;
    let throughput_ghs = if elapsed.as_secs_f64() > 0.0 {
        (searched as f64 / elapsed.as_secs_f64()) / 1.0e9
    } else {
        0.0
    };

    let (hash_hex, best_lz_cpu) = double_sha256_lz(&params.outpoint, best_nonce);

    if best_hash != hash_hex {
        println!(
            "warning: gpu hash {} differs from cpu hash {} for outpoint {} nonce {}",
            best_hash, hash_hex, params.outpoint, best_nonce
        );
    }

    if best_lz_cpu != best_lz_gpu {
        println!(
            "warning: gpu lz {} differs from cpu lz {} for outpoint {} nonce {}",
            best_lz_gpu, best_lz_cpu, params.outpoint, best_nonce
        );
    }

    Ok(JobResult {
        outpoint: params.outpoint.clone(),
        best_nonce: best_nonce.to_string(),
        best_lz: best_lz_cpu,
        best_hash: hash_hex,
        searched,
        start_nonce: params.start_nonce.to_string(),
        duration_ms,
        throughput_ghs,
    })
}

fn double_sha256_lz(challenge: &str, nonce: u64) -> (String, u32) {
    let nonce_str = nonce.to_string();
    let mut message = Vec::with_capacity(challenge.len() + nonce_str.len());
    message.extend_from_slice(challenge.as_bytes());
    message.extend_from_slice(nonce_str.as_bytes());

    let mut hasher = Sha256::new();
    hasher.update(&message);
    let first = hasher.finalize_reset();
    hasher.update(&first);
    let second = hasher.finalize();

    let hash_hex = hex_encode(&second);
    let mut lz = 0u32;
    for byte in second.iter() {
        if *byte == 0 {
            lz += 8;
            continue;
        }
        for bit in (0..8).rev() {
            if (byte >> bit) & 1 == 1 {
                return (hash_hex, lz);
            }
            lz += 1;
        }
        break;
    }
    (hash_hex, lz)
}
