use std::io;

use serde::Serialize;

#[cfg(not(any(feature = "cuda", feature = "fpga")))]
compile_error!("bro crate requires enabling at least one of the 'cuda' or 'fpga' features");

#[cfg(feature = "cuda")]
mod cuda_backend;
#[cfg(feature = "fpga")]
mod fpga_backend;

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    #[cfg(feature = "fpga")]
    Fpga,
}

impl BackendKind {
    pub fn from_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Some(Self::Cuda),
            #[cfg(feature = "fpga")]
            "fpga" | "aws-f2" | "awsf2" | "aws_f2" => Some(Self::Fpga),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            #[cfg(feature = "fpga")]
            Self::Fpga => "fpga",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub backend: BackendKind,
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
    #[cfg(feature = "fpga")]
    pub fpga: Option<FpgaRuntimeConfig>,
}

impl SearchConfig {
    pub fn with_outpoint(outpoint: String) -> Self {
        let default_backend = {
            #[cfg(feature = "cuda")]
            {
                BackendKind::Cuda
            }
            #[cfg(all(not(feature = "cuda"), feature = "fpga"))]
            {
                BackendKind::Fpga
            }
        };

        Self {
            backend: default_backend,
            outpoint,
            total_nonce_all: DEFAULT_TOTAL_NONCE.min(MAX_TOTAL_NONCE),
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
            #[cfg(feature = "fpga")]
            fpga: None,
        }
    }

    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }
}

pub const DEFAULT_TOTAL_NONCE: u64 = 100_000_000_000_000;
pub const DEFAULT_START_NONCE: u64 = 0;
pub const DEFAULT_BATCH_SIZE: u64 = 1_000_000_000;
pub const DEFAULT_THREADS_PER_BLOCK: u32 = 256;
pub const DEFAULT_BLOCKS: u32 = 1024;
pub const DEFAULT_CHUNK_SIZE: u32 = 65_536;
pub const DEFAULT_ILP: u32 = 1;
pub const MAX_ILP: u32 = 8;
pub const DEFAULT_PROGRESS_MS: u64 = 0;
pub const DEFAULT_ODOMETER: bool = true;
pub const MAX_TOTAL_NONCE: u64 = 2_000_000_000_000;

#[cfg(feature = "fpga")]
#[derive(Clone, Debug)]
pub struct FpgaRuntimeConfig {
    pub slot_id: u32,
    pub xclbin_path: Option<String>,
    pub dma_buffer_bytes: u32,
    pub batches_per_exec: u32,
    pub streams: u32,
}

#[cfg(feature = "fpga")]
impl Default for FpgaRuntimeConfig {
    fn default() -> Self {
        Self {
            slot_id: 0,
            xclbin_path: None,
            dma_buffer_bytes: 4 * 1024 * 1024,
            batches_per_exec: 8192,
            streams: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchOutcome {
    pub best_lz: u32,
    pub best_nonce: u64,
    pub elapsed_secs: f64,
    pub rate_ghs: f64,
    pub total_nonce: u64,
    pub accelerator_count: usize,
}

pub fn run_search(config: SearchConfig) -> Result<SearchOutcome, DynError> {
    match config.backend {
        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_backend::run_search_cuda(config)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Box::new(io::Error::new(
                    io::ErrorKind::Other,
                    "CUDA backend not compiled in",
                )))
            }
        }
        #[cfg(feature = "fpga")]
        BackendKind::Fpga => {
            #[cfg(feature = "fpga")]
            {
                fpga_backend::run_search_fpga(config)
            }
            #[cfg(not(feature = "fpga"))]
            {
                Err(Box::new(io::Error::new(
                    io::ErrorKind::Other,
                    "FPGA backend not compiled in",
                )))
            }
        }
    }
}
