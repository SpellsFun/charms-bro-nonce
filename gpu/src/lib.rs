#[cfg(all(feature = "cuda", feature = "stub"))]
compile_error!("features `cuda` and `stub` are mutually exclusive");

#[cfg(not(any(feature = "cuda", feature = "stub")))]
compile_error!("enable either the `cuda` or `stub` feature for the gpu crate");

#[cfg(feature = "cuda")]
mod cuda {
    use anyhow::{ensure, Result};
    use cust::{memory::*, prelude::*};
    use hex::encode as hex_encode;

    pub struct Miner {
        _ctx: Context, // 保持上下文存活
        module: Module,
        stream: Stream,
    }

    impl Miner {
        pub fn new() -> Result<Self> {
            // 选第0块卡；如需多卡自行扩展
            let _ctx = cust::quick_init()?;
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));
            let module = Module::from_ptx(ptx, &[])?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            Ok(Self {
                _ctx,
                module,
                stream,
            })
        }

        /// 运行一次批量搜索：从 start_nonce 起，count 个
        #[allow(clippy::too_many_arguments)]
        pub fn mine_batch(
            &self,
            challenge: &[u8],
            start_nonce: u64,
            count: u32,
            blocks: u32,
            threads_per_block: u32,
            ilp: u32,
        ) -> Result<(u64, u32, String)> {
            ensure!(count > 0, "count must be greater than zero");
            ensure!(blocks > 0, "blocks must be greater than zero");
            ensure!(
                threads_per_block > 0,
                "threads_per_block must be greater than zero"
            );
            ensure!(ilp > 0, "ilp must be greater than zero");

            // 设备内存准备
            let d_challenge = DeviceBuffer::from_slice(challenge)?;
            let d_best_digest = DeviceBuffer::<u32>::zeroed(8)?; // 8 * u32
            let d_best_info = DeviceBuffer::<u32>::zeroed(4)?; // [lz, lo, hi, lock]

            // 换一种方式：获取 "launch_mine" host wrapper 并调用
            let func = self.module.get_function("launch_mine")?;
            // 注意：通过 C wrapper 传指针
            let clen = challenge.len() as u32;
            let blocks_i32 =
                i32::try_from(blocks).map_err(|_| anyhow::anyhow!("blocks exceeds i32"))?;
            let threads_i32 = i32::try_from(threads_per_block)
                .map_err(|_| anyhow::anyhow!("threads_per_block exceeds i32"))?;

            let mut stream_ptr = self.stream.as_inner();
            let mut challenge_ptr = d_challenge.as_device_ptr().as_raw();
            let mut clen_mut = clen;
            let mut start_nonce_mut = start_nonce;
            let mut count_mut = count;
            let mut ilp_mut = ilp;
            let mut digest_ptr = d_best_digest.as_device_ptr().as_raw();
            let mut info_ptr = d_best_info.as_device_ptr().as_raw();
            let mut blocks_mut = blocks_i32;
            let mut threads_mut = threads_i32;
            let args = [
                &mut stream_ptr as *mut _ as *mut std::ffi::c_void,
                &mut challenge_ptr as *mut _ as *mut std::ffi::c_void,
                &mut clen_mut as *mut _ as *mut std::ffi::c_void,
                &mut start_nonce_mut as *mut _ as *mut std::ffi::c_void,
                &mut count_mut as *mut _ as *mut std::ffi::c_void,
                &mut ilp_mut as *mut _ as *mut std::ffi::c_void,
                &mut digest_ptr as *mut _ as *mut std::ffi::c_void,
                &mut info_ptr as *mut _ as *mut std::ffi::c_void,
                &mut blocks_mut as *mut _ as *mut std::ffi::c_void,
                &mut threads_mut as *mut _ as *mut std::ffi::c_void,
            ];

            unsafe {
                self.stream.launch(&func, (1, 1, 1), (1, 1, 1), 0, &args)?;
            }
            self.stream.synchronize()?;

            // 拷回结果
            let mut best_info = [0u32; 4];
            let mut best_digest = [0u32; 8];
            d_best_info.copy_to(&mut best_info)?;
            d_best_digest.copy_to(&mut best_digest)?;

            let lz = best_info[0];
            let nonce = (best_info[2] as u64) << 32 | (best_info[1] as u64);

            // 组装 hash hex（大端 u32 串联）
            let mut bytes = [0u8; 32];
            for i in 0..8 {
                // 把每个 u32 按大端写入
                let w = best_digest[i].to_be_bytes();
                bytes[i * 4..i * 4 + 4].copy_from_slice(&w);
            }
            let hash_hex = hex_encode(bytes);

            Ok((nonce, lz, hash_hex))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::Miner;

#[cfg(feature = "stub")]
mod stub {
    use anyhow::{anyhow, Result};

    pub struct Miner;

    impl Miner {
        pub fn new() -> Result<Self> {
            Err(anyhow!(
                "gpu crate was built without the `cuda` feature; rebuild with `--no-default-features --features gpu/cuda` on a CUDA-capable host"
            ))
        }

        #[allow(clippy::unused_self)]
        #[allow(clippy::too_many_arguments)]
        pub fn mine_batch(
            &self,
            _challenge: &[u8],
            _start_nonce: u64,
            _count: u32,
            _blocks: u32,
            _threads_per_block: u32,
            _ilp: u32,
        ) -> Result<(u64, u32, String)> {
            Err(anyhow!(
                "CUDA backend unavailable; rebuild gpu crate with the `cuda` feature on a machine with NVIDIA drivers"
            ))
        }
    }
}

#[cfg(feature = "stub")]
pub use stub::Miner;
