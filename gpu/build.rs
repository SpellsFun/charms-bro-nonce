use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    // 如果未启用 cuda 特性，直接跳过编译 PTX
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        println!("cargo:warning=CUDA feature disabled; skipping kernel compilation");
        println!("cargo:rerun-if-changed=kernel.cu");
        return;
    }

    // 目标 PTX 路径
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("kernel.ptx");
    let cu_path = PathBuf::from("kernel.cu"); // 相对 gpu/ 目录

    // 允许通过环境变量覆盖 nvcc 路径或架构
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let arch = env::var("CUDA_ARCH")
        .or_else(|_| env::var("ARCH"))
        .unwrap_or_else(|_| "sm_89".to_string());

    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_RREG");
    println!("cargo:rerun-if-env-changed=RREG");
    println!("cargo:rerun-if-env-changed=NVCC_FLAGS");

    let mut command = Command::new(&nvcc);
    command.arg("-ptx");
    command.arg("-arch");
    command.arg(&arch);
    command.arg(cu_path.to_str().unwrap());
    command.arg("-o");
    command.arg(ptx_path.to_str().unwrap());
    command.arg("-O3");
    command.arg("--use_fast_math");

    if let Ok(rreg) = env::var("CUDA_RREG").or_else(|_| env::var("RREG")) {
        if !rreg.trim().is_empty() {
            command.arg("-maxrregcount");
            command.arg(rreg.trim());
        }
    }

    if let Ok(extra) = env::var("NVCC_FLAGS") {
        for flag in extra.split_whitespace() {
            if !flag.is_empty() {
                command.arg(flag);
            }
        }
    }

    let status = command
        .status()
        .unwrap_or_else(|err| {
            panic!(
                "Failed to run nvcc (command `{}`): {}. Ensure CUDA Toolkit is installed and NVCC is on PATH or set NVCC env var.",
                nvcc, err
            )
        });

    if !status.success() {
        panic!(
            "nvcc (command `{}`) exited with code {:?} while compiling {:?}",
            nvcc,
            status.code(),
            cu_path
        );
    }

    // 把 PTX 内容写到 rerun 指令里以便缓存失效
    println!("cargo:rerun-if-changed=kernel.cu");

    // 也可以把 ptx 复制到项目根用于调试（可选）
    let _ = fs::copy(&ptx_path, PathBuf::from("kernel.ptx"));
}
