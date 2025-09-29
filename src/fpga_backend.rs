use crate::{DynError, FpgaRuntimeConfig, SearchConfig, SearchOutcome};
use chrono::Local;
use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device};
use opencl3::kernel::Kernel;
use opencl3::memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_device_id, CL_DEVICE_TYPE_ACCELERATOR};
use std::fs;
use std::io;
use std::time::Instant;

pub fn run_search_fpga(mut config: SearchConfig) -> Result<SearchOutcome, DynError> {
    if config.total_nonce_all == 0 {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "TOTAL_NONCE must be greater than 0",
        )));
    }

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

    let fpga_cfg = config
        .fpga
        .clone()
        .unwrap_or_else(FpgaRuntimeConfig::default);
    let xclbin_path = fpga_cfg.xclbin_path.clone().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "FPGA backend requires fpga.xclbin_path to be specified",
        )
    })?;

    // Discover accelerator devices on the system
    let devices = get_all_devices(CL_DEVICE_TYPE_ACCELERATOR)?;
    if devices.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            "no OpenCL accelerator devices detected (ensure aws-fpga runtime is installed)",
        )));
    }
    let slot_index = fpga_cfg.slot_id as usize;
    let device_id: cl_device_id = *devices.get(slot_index).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "requested FPGA slot {} but only {} accelerator device(s) found",
                slot_index,
                devices.len()
            ),
        )
    })?;
    let device = Device::new(device_id);
    let device_name = device.name()?;

    println!(
        "[{}] Using FPGA device {} ({})",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        slot_index,
        device_name
    );

    // Build OpenCL context and command queue
    let context = Context::from_device(&device)?;
    let queue = CommandQueue::create_default(&context, device_id)?;

    // Load precompiled xclbin
    let binary = fs::read(&xclbin_path)?;
    let program = Program::create_from_binary(&context, &[device_id], &[&binary])?;
    program.build(&[])?;
    let mut kernel = Kernel::create(&program, "double_sha256_fpga")?;

    let base_bytes = config.outpoint.clone().into_bytes();
    let base_buffer = Buffer::<u8>::create(
        &context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        base_bytes.len(),
        Some(&base_bytes),
    )?;
    let best_lz_buffer = Buffer::<u32>::create(&context, CL_MEM_READ_WRITE, 1, None)?;
    let best_nonce_buffer = Buffer::<u64>::create(&context, CL_MEM_READ_WRITE, 1, None)?;

    // Kernel arguments constant across launches
    kernel.set_arg(0, &base_buffer)?;
    kernel.set_arg(1, &(base_bytes.len() as u32))?;
    kernel.set_arg(4, &(if config.binary_nonce { 1u32 } else { 0u32 }))?;
    kernel.set_arg(5, &best_lz_buffer)?;
    kernel.set_arg(6, &best_nonce_buffer)?;

    let t0 = Instant::now();
    let mut best_lz = 0u32;
    let mut best_nonce = 0u64;
    let mut processed = 0u64;
    let chunk = fpga_cfg.batches_per_exec.max(1) as u64;
    let total = config.total_nonce_all;

    while processed < total {
        let remaining = total - processed;
        let work = chunk.min(remaining);
        let start_nonce = config.start_nonce_all + processed;

        let zero_u32 = [0u32];
        let zero_u64 = [0u64];
        queue.enqueue_write_buffer(&best_lz_buffer, CL_BLOCKING, 0, &zero_u32, &[])?;
        queue.enqueue_write_buffer(&best_nonce_buffer, CL_BLOCKING, 0, &zero_u64, &[])?;

        kernel.set_arg(2, &(start_nonce as u64))?;
        kernel.set_arg(3, &(work as u64))?;

        let global_work = [1usize];
        queue.enqueue_nd_range_kernel(&kernel, 1, None, &global_work, None, &[])?;
        queue.finish()?;

        let mut batch_lz = vec![0u32; 1];
        let mut batch_nonce = vec![0u64; 1];
        queue.enqueue_read_buffer(&best_lz_buffer, CL_BLOCKING, 0, &mut batch_lz, &[])?;
        queue.enqueue_read_buffer(&best_nonce_buffer, CL_BLOCKING, 0, &mut batch_nonce, &[])?;

        if batch_lz[0] > best_lz {
            best_lz = batch_lz[0];
            best_nonce = batch_nonce[0];
        }

        processed += work;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let ghps = if elapsed > 0.0 {
        (processed as f64) / elapsed / 1e9
    } else {
        0.0
    };

    println!(
        "[{}] FPGA search done: best_lz={} nonce={} processed={} elapsed={:.2}s ({:.2} GH/s)",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        best_lz,
        best_nonce,
        processed,
        elapsed,
        ghps
    );

    Ok(SearchOutcome {
        best_lz,
        best_nonce,
        elapsed_secs: elapsed,
        rate_ghs: ghps,
        total_nonce: processed,
        accelerator_count: fpga_cfg.streams.max(1) as usize,
    })
}
