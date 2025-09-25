use rustacuda::context::CacheConfig;
use rustacuda::launch;
use rustacuda::memory::*;
use rustacuda::prelude::*;
use serde::Serialize;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::mem;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};

static STOP: AtomicBool = AtomicBool::new(false);

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Debug)]
struct GpuConfig {
    start_nonce: u64,
    total_nonce: u64,
    batch_size: u64,
    threads_per_block: u32,
    blocks: u32,
    binary_nonce: u32,
    persistent: bool,
    chunk_size: u32,
    ilp: u32,
    progress_ms: u64,
    odometer: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct GpuResult {
    best_lz: u32,
    best_nonce: u64,
}

#[derive(Debug)]
struct SharedState {
    done: AtomicU64,
    best_lz: AtomicU32,
    best_nonce: AtomicU64,
    finished: AtomicBool,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub outpoint: String,
    pub total_nonce_all: u64,
    pub start_nonce_all: u64,
    pub batch_size: u64,
    pub threads_per_block: u32,
    pub blocks: u32,
    pub binary_nonce: bool,
    pub persistent: bool,
    pub chunk_size: u32,
    pub ilp: u32,
    pub progress_ms: u64,
    pub odometer: bool,
    pub gpu_ids: Option<Vec<u32>>,
    pub gpu_weights: Option<Vec<f64>>,
}

impl SearchConfig {
    pub fn with_outpoint(outpoint: String) -> Self {
        Self {
            outpoint,
            total_nonce_all: DEFAULT_TOTAL_NONCE,
            start_nonce_all: DEFAULT_START_NONCE,
            batch_size: DEFAULT_BATCH_SIZE,
            threads_per_block: DEFAULT_THREADS_PER_BLOCK,
            blocks: DEFAULT_BLOCKS,
            binary_nonce: false,
            persistent: false,
            chunk_size: DEFAULT_CHUNK_SIZE,
            ilp: DEFAULT_ILP,
            progress_ms: DEFAULT_PROGRESS_MS,
            odometer: DEFAULT_ODOMETER,
            gpu_ids: None,
            gpu_weights: None,
        }
    }
}

pub const DEFAULT_TOTAL_NONCE: u64 = 100_000_000_000_000;
pub const DEFAULT_START_NONCE: u64 = 0;
pub const DEFAULT_BATCH_SIZE: u64 = 10_000_000_000;
pub const DEFAULT_THREADS_PER_BLOCK: u32 = 512;
pub const DEFAULT_BLOCKS: u32 = 1024;
pub const DEFAULT_CHUNK_SIZE: u32 = 131_072;
pub const DEFAULT_ILP: u32 = 8;
pub const MAX_ILP: u32 = 32;
pub const DEFAULT_PROGRESS_MS: u64 = 0;
pub const DEFAULT_ODOMETER: bool = true;

#[derive(Clone, Debug, Serialize)]
pub struct SearchOutcome {
    pub best_lz: u32,
    pub best_nonce: u64,
    pub elapsed_secs: f64,
    pub rate_ghs: f64,
    pub total_nonce: u64,
    pub gpu_count: usize,
}

pub fn run_search(config: SearchConfig) -> Result<SearchOutcome, DynError> {
    STOP.store(false, Ordering::SeqCst);

    let mut config = config;
    config.ilp = config.ilp.clamp(1, MAX_ILP);

    let monitor_interval = config.progress_ms;
    let user_requested_progress = monitor_interval > 0;

    let base_len = config.outpoint.as_bytes().len();
    let reserve = if config.binary_nonce { 8 } else { 20 };
    if base_len + reserve > 128 {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "outpoint string too long: len={} (needs len + {} <= 128)",
                base_len, reserve
            ),
        )));
    }

    rustacuda::init(CudaFlags::empty())?;
    let total_devices = Device::num_devices()? as u32;
    if total_devices == 0 {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            "no CUDA devices detected",
        )));
    }

    let mut gpu_indices = if let Some(ids) = config.gpu_ids.clone() {
        let mut ids = ids;
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            vec![0]
        } else {
            for &id in &ids {
                if id >= total_devices {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "requested GPU index {} but only {} device(s) detected",
                            id, total_devices
                        ),
                    )));
                }
            }
            ids
        }
    } else {
        (0..total_devices).collect()
    };
    if gpu_indices.is_empty() {
        gpu_indices.push(0);
    }

    let weights = if let Some(ws) = config.gpu_weights.clone() {
        if ws.len() != gpu_indices.len() {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "GPU weight count ({}) does not match GPU count ({})",
                    ws.len(),
                    gpu_indices.len()
                ),
            )));
        }
        ws
    } else {
        vec![1.0; gpu_indices.len()]
    };

    println!(
        "Launching {} GPU worker(s): {:?}",
        gpu_indices.len(),
        gpu_indices
    );
    for &gpu in &gpu_indices {
        let dev = Device::get_device(gpu)?;
        let name = dev.name()?;
        println!("  - GPU {}: {}", gpu, name);
    }

    if config.total_nonce_all == 0 {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "TOTAL_NONCE must be greater than 0",
        )));
    }

    let mut tasks: Vec<(u32, GpuConfig)> = Vec::new();
    let sum_w = weights.iter().copied().sum::<f64>().max(1e-9);
    let mut acc = 0.0_f64;
    for (i, &gpu) in gpu_indices.iter().enumerate() {
        let w = weights[i].max(0.0);
        let start_off = ((config.total_nonce_all as f64) * acc / sum_w).floor() as u64;
        acc += w;
        let end_off = ((config.total_nonce_all as f64) * acc / sum_w).floor() as u64;
        let len = end_off.saturating_sub(start_off);
        let start = config.start_nonce_all + start_off;
        if len == 0 {
            continue;
        }
        let gpu_cfg = GpuConfig {
            start_nonce: start,
            total_nonce: len,
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            blocks: config.blocks,
            binary_nonce: config.binary_nonce as u32,
            persistent: config.persistent,
            chunk_size: config.chunk_size,
            ilp: config.ilp,
            progress_ms: config.progress_ms,
            odometer: config.odometer as u32,
        };
        tasks.push((gpu, gpu_cfg));
    }

    if tasks.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no work scheduled for any GPU (check TOTAL_NONCE and GPU weights)",
        )));
    }

    let assigned_gpus: Vec<u32> = tasks.iter().map(|(gpu, _)| *gpu).collect();
    let gpu_count = assigned_gpus.len();

    let t0 = Instant::now();
    let fixed_bytes = config.outpoint.into_bytes();

    let mut handles = Vec::new();
    let mut shared_states: Vec<Arc<SharedState>> = Vec::new();
    let mut gpu_t0: Vec<Instant> = Vec::new();

    for (gpu, cfg) in tasks.into_iter() {
        let bytes = fixed_bytes.clone();
        let st = Arc::new(SharedState {
            done: AtomicU64::new(0),
            best_lz: AtomicU32::new(0),
            best_nonce: AtomicU64::new(0),
            finished: AtomicBool::new(false),
        });
        gpu_t0.push(Instant::now());
        let st_for_err = st.clone();
        shared_states.push(st.clone());
        handles.push(thread::spawn(move || -> Result<(u32, u64), DynError> {
            match run_on_device(gpu, &bytes, &cfg, Some(st)) {
                Ok(res) => Ok((res.best_lz, res.best_nonce)),
                Err(e) => {
                    st_for_err.finished.store(true, Ordering::SeqCst);
                    Err(Box::new(io::Error::new(
                        io::ErrorKind::Other,
                        e.to_string(),
                    )))
                }
            }
        }));
    }

    if monitor_interval > 0 {
        let total_all = config.total_nonce_all;
        let inline = io::stdout().is_terminal();
        loop {
            thread::sleep(Duration::from_millis(monitor_interval));
            let mut sum_done = 0u64;
            let mut rates: Vec<f64> = Vec::with_capacity(shared_states.len());
            for (i, st) in shared_states.iter().enumerate() {
                let done = st.done.load(Ordering::Relaxed);
                sum_done = sum_done.saturating_add(done);
                let secs = gpu_t0[i].elapsed().as_secs_f64();
                let r = if secs > 0.0 {
                    (done as f64) / secs / 1e9
                } else {
                    0.0
                };
                rates.push(r);
            }
            let pct = if total_all > 0 {
                (sum_done as f64) / (total_all as f64) * 100.0
            } else {
                0.0
            };
            let mut g_best_lz = 0u32;
            let mut g_best_nonce = 0u64;
            for st in &shared_states {
                let lz = st.best_lz.load(Ordering::Relaxed);
                let no = st.best_nonce.load(Ordering::Relaxed);
                if lz > g_best_lz {
                    g_best_lz = lz;
                    g_best_nonce = no;
                }
            }

            if user_requested_progress {
                let mut line = format!("Progress: {:.2}% |", pct.min(100.0));
                for (i, r) in rates.iter().enumerate() {
                    if i > 0 {
                        line.push(' ');
                    }
                    line.push_str(&format!(" GPU{} {:.2} GH/s", assigned_gpus[i], r));
                    if i + 1 < rates.len() {
                        line.push(',');
                    }
                }
                line.push_str(&format!(" | best_lz={} nonce={}", g_best_lz, g_best_nonce));
                if inline {
                    print!("\r{}", line);
                    let _ = io::stdout().flush();
                } else {
                    println!("{}", line);
                }
            }

            if shared_states
                .iter()
                .all(|s| s.finished.load(Ordering::SeqCst))
            {
                break;
            }
        }
        if user_requested_progress && inline {
            println!();
        }
    }

    let mut best_lz = 0u32;
    let mut best_nonce = 0u64;
    for (idx, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok((lz, nonce))) => {
                println!(
                    "[GPU {}] done: best_lz={} nonce={}",
                    assigned_gpus[idx], lz, nonce
                );
                if lz > best_lz {
                    best_lz = lz;
                    best_nonce = nonce;
                }
            }
            Ok(Err(e)) => {
                return Err(e);
            }
            Err(e) => {
                return Err(Box::new(io::Error::new(
                    io::ErrorKind::Other,
                    format!("GPU worker panicked: {e:?}"),
                )));
            }
        }
    }

    let total_processed: u64 = shared_states
        .iter()
        .map(|st| st.done.load(Ordering::Relaxed))
        .sum();
    let elapsed = t0.elapsed().as_secs_f64();
    let ghps = if elapsed > 0.0 {
        (total_processed as f64) / elapsed / 1e9
    } else {
        0.0
    };
    println!(
        "Final best leading zeros: {} at nonce {}",
        best_lz, best_nonce
    );
    println!(
        "Summary: GPUs={} elapsed {:.2}s, rate {:.2} GH/s",
        gpu_count, elapsed, ghps
    );

    Ok(SearchOutcome {
        best_lz,
        best_nonce,
        elapsed_secs: elapsed,
        rate_ghs: ghps,
        total_nonce: total_processed,
        gpu_count,
    })
}

fn run_on_device(
    device_idx: u32,
    fixed: &[u8],
    cfg: &GpuConfig,
    shared: Option<Arc<SharedState>>,
) -> Result<GpuResult, DynError> {
    let device = Device::get_device(device_idx)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let module = if Path::new("sha256_kernel.cubin").exists() {
        match Module::load_from_file(&CString::new("sha256_kernel.cubin")?) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "[GPU {}] load cubin failed: {}. Falling back to PTX.",
                    device_idx, e
                );
                if Path::new("sha256_kernel.ptx").exists() {
                    let ptx = fs::read_to_string("sha256_kernel.ptx")?;
                    Module::load_from_string(&CString::new(ptx)?)?
                } else {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::NotFound,
                        "kernel module not found: run ./build_cubin_ada.sh to build sha256_kernel.cubin or sha256_kernel.ptx",
                    )));
                }
            }
        }
    } else if Path::new("sha256_kernel.ptx").exists() {
        let ptx = fs::read_to_string("sha256_kernel.ptx")?;
        Module::load_from_string(&CString::new(ptx)?)?
    } else {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            "kernel module not found: run ./build_cubin_ada.sh to build sha256_kernel.cubin or sha256_kernel.ptx",
        )));
    };

    let mut func =
        module.get_function(CStr::from_bytes_with_nul(b"double_sha256_max_kernel\0")?)?;
    let mut persistent_func = if cfg.binary_nonce == 0 {
        module.get_function(CStr::from_bytes_with_nul(
            b"double_sha256_persistent_kernel_ascii\0",
        )?)?
    } else {
        module.get_function(CStr::from_bytes_with_nul(
            b"double_sha256_persistent_kernel\0",
        )?)?
    };
    let reduce_func = module.get_function(CStr::from_bytes_with_nul(b"reduce_best_kernel\0")?)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let progress_stream = if cfg.progress_ms > 0 {
        Some(Stream::new(StreamFlags::NON_BLOCKING, None)?)
    } else {
        None
    };

    let _ = func.set_cache_config(CacheConfig::PreferL1);
    let _ = persistent_func.set_cache_config(CacheConfig::PreferL1);

    let mut d_base = DeviceBuffer::from_slice(fixed)?;
    let mut d_block_lz = unsafe { DeviceBuffer::<u32>::zeroed(cfg.blocks as usize)? };
    let mut d_block_nonce = unsafe { DeviceBuffer::<u64>::zeroed(cfg.blocks as usize)? };
    let mut d_best_lz = DeviceBox::new(&0u32)?;
    let mut d_best_nonce = DeviceBox::new(&0u64)?;
    let mut d_best_lz_live = DeviceBox::new(&0u32)?;
    let mut d_best_nonce_live = DeviceBox::new(&0u64)?;
    let mut d_stop_flag = DeviceBox::new(&0u32)?;
    let mut d_next_index = DeviceBox::new(&0u64)?;

    let mut best_lz = 0u32;
    let mut best_nonce = 0u64;
    let base_len = fixed.len();

    let num_batches = if cfg.persistent {
        1
    } else if cfg.batch_size == 0 {
        1
    } else {
        (cfg.total_nonce + cfg.batch_size - 1) / cfg.batch_size
    };

    unsafe {
        if cfg.persistent {
            d_next_index.copy_from(&0u64)?;
            d_best_lz_live.copy_from(&0u32)?;
            d_best_nonce_live.copy_from(&0u64)?;
            d_stop_flag.copy_from(&0u32)?;

            let warps = (cfg.threads_per_block + 31) / 32;
            let base_rem = (base_len as u32) % 64;
            let shmem_bytes = (warps * (mem::size_of::<u32>() + mem::size_of::<u64>()) as u32)
                + (8 * mem::size_of::<u32>() as u32)
                + base_rem;
            let enable_live: u32 = if cfg.progress_ms > 0 { 1 } else { 0 };
            launch!(persistent_func<<<cfg.blocks, cfg.threads_per_block, shmem_bytes, stream>>>(
                d_base.as_device_ptr(),
                base_len,
                cfg.start_nonce,
                cfg.total_nonce,
                cfg.binary_nonce,
                d_next_index.as_device_ptr(),
                cfg.chunk_size,
                cfg.ilp,
                enable_live,
                cfg.odometer,
                d_block_nonce.as_device_ptr(),
                d_block_lz.as_device_ptr(),
                d_best_lz_live.as_device_ptr(),
                d_best_nonce_live.as_device_ptr(),
                d_stop_flag.as_device_ptr()
            ))?;

            if let Some(_prog_stream) = &progress_stream {
                let mut last = 0u64;
                loop {
                    sleep(Duration::from_millis(cfg.progress_ms));
                    if STOP.load(Ordering::SeqCst) {
                        d_stop_flag.copy_from(&1u32)?;
                        break;
                    }
                    let mut done: u64 = 0;
                    d_next_index.copy_to(&mut done)?;
                    if let Some(st) = &shared {
                        if done != last {
                            let mut live_lz: u32 = 0;
                            let mut live_nonce: u64 = 0;
                            d_best_lz_live.copy_to(&mut live_lz)?;
                            d_best_nonce_live.copy_to(&mut live_nonce)?;
                            st.done.store(done, Ordering::Relaxed);
                            st.best_lz.store(live_lz, Ordering::Relaxed);
                            st.best_nonce.store(live_nonce, Ordering::Relaxed);
                            last = done;
                        }
                    }
                    if done >= cfg.total_nonce {
                        break;
                    }
                }
            }

            stream.synchronize()?;

            let mut final_done = cfg.total_nonce;
            d_next_index.copy_to(&mut final_done)?;
            if final_done > cfg.total_nonce {
                final_done = cfg.total_nonce;
            }

            let reduce_threads = {
                let mut t = 1u32;
                while t < cfg.blocks && t < 1024 {
                    t <<= 1;
                }
                if t > 1024 {
                    1024
                } else {
                    t
                }
            };
            let reduce_warps = (reduce_threads + 31) / 32;
            let reduce_shmem =
                reduce_warps * (mem::size_of::<u32>() + mem::size_of::<u64>()) as u32;
            launch!(reduce_func<<<1, reduce_threads, reduce_shmem, stream>>>(
                d_block_lz.as_device_ptr(),
                d_block_nonce.as_device_ptr(),
                cfg.blocks,
                d_best_lz.as_device_ptr(),
                d_best_nonce.as_device_ptr()
            ))?;

            stream.synchronize()?;

            d_best_lz.copy_to(&mut best_lz)?;
            d_best_nonce.copy_to(&mut best_nonce)?;

            if let Some(st) = &shared {
                st.done.store(final_done, Ordering::Relaxed);
            }
        } else {
            for batch_idx in 0..num_batches {
                if STOP.load(Ordering::SeqCst) {
                    break;
                }
                let (start_rel, batch_nonce) = if cfg.batch_size == 0 {
                    (0u64, cfg.total_nonce)
                } else {
                    let start = batch_idx * cfg.batch_size;
                    let bn = std::cmp::min(cfg.batch_size, cfg.total_nonce - start);
                    (start, bn)
                };
                let start_nonce = cfg.start_nonce + start_rel;

                let warps = (cfg.threads_per_block + 31) / 32;
                let base_rem = (base_len as u32) % 64;
                let shmem_bytes = (warps * (mem::size_of::<u32>() + mem::size_of::<u64>()) as u32)
                    + (8 * mem::size_of::<u32>() as u32)
                    + base_rem;
                launch!(func<<<cfg.blocks, cfg.threads_per_block, shmem_bytes, stream>>>(
                    d_base.as_device_ptr(),
                    base_len,
                    start_nonce,
                    batch_nonce,
                    cfg.binary_nonce,
                    d_block_nonce.as_device_ptr(),
                    d_block_lz.as_device_ptr()
                ))?;

                let reduce_threads = {
                    let mut t = 1u32;
                    while t < cfg.blocks && t < 1024 {
                        t <<= 1;
                    }
                    if t > 1024 {
                        1024
                    } else {
                        t
                    }
                };
                let reduce_warps = (reduce_threads + 31) / 32;
                let reduce_shmem =
                    reduce_warps * (mem::size_of::<u32>() + mem::size_of::<u64>()) as u32;
                launch!(reduce_func<<<1, reduce_threads, reduce_shmem, stream>>>(
                    d_block_lz.as_device_ptr(),
                    d_block_nonce.as_device_ptr(),
                    cfg.blocks,
                    d_best_lz.as_device_ptr(),
                    d_best_nonce.as_device_ptr()
                ))?;

                stream.synchronize()?;

                let mut batch_lz: u32 = 0;
                let mut batch_nonce_best: u64 = 0;
                d_best_lz.copy_to(&mut batch_lz)?;
                d_best_nonce.copy_to(&mut batch_nonce_best)?;
                if batch_lz > best_lz {
                    best_lz = batch_lz;
                    best_nonce = batch_nonce_best;
                }

                if batch_idx % 10 == 0 || batch_idx + 1 == num_batches {
                    println!(
                        "[GPU {}] Batch {} done, current best_lz={} nonce={} current={}",
                        device_idx, batch_idx, best_lz, best_nonce, start_nonce
                    );
                }

                if let Some(st) = &shared {
                    let completed = start_rel.saturating_add(batch_nonce);
                    st.done.store(completed, Ordering::Relaxed);
                    st.best_lz.store(best_lz, Ordering::Relaxed);
                    st.best_nonce.store(best_nonce, Ordering::Relaxed);
                }
            }
        }
    }

    if let Some(st) = &shared {
        st.finished.store(true, Ordering::SeqCst);
        let _ = st.best_lz.fetch_max(best_lz, Ordering::Relaxed);
        if st.best_lz.load(Ordering::Relaxed) == best_lz {
            st.best_nonce.store(best_nonce, Ordering::Relaxed);
        }
    }

    Ok(GpuResult {
        best_lz,
        best_nonce,
    })
}
