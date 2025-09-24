# charms-bro-nonce

用于挖矿 PoW 计算的 Rust + CUDA 服务端，包含一个 GPU 计算内核 (`gpu/`) 和一个对外的 HTTP 服务 (`server/`)。

## 项目结构
- `gpu/`：CUDA kernel 与通过 [`cust`](https://crates.io/crates/cust) 暴露的 Rust 封装。
- `server/`：基于 Axum 的 HTTP API，调度 GPU 批量计算。
- `Cargo.toml`：工作区根，统一管理 `server` 与 `gpu` 两个 crate。

## 运行环境与前置依赖
> 当前实现需真实的 NVIDIA GPU 与 CUDA 驱动，macOS (Apple 芯片) 上无法直接编译/运行，可通过 Linux GPU 服务器或支持 NVIDIA GPU 的容器环境来部署。

1. **NVIDIA 驱动**：建议与目标 GPU 匹配的最新稳定版本。
2. **CUDA Toolkit 12.x+**：需要 `nvcc`、CUDA Driver API 与 NVRTC。安装完成后请确认 `nvcc --version` 能正常执行。
3. **Rust toolchain 1.75+**：推荐通过 `rustup` 安装：
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   ```
4. **环境变量**：确保以下变量在构建时可用（示例以默认安装路径 `/usr/local/cuda` 为准）：
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH="$CUDA_HOME/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
   ```
   - `NVCC=/path/to/nvcc`：覆盖默认的 nvcc 路径。
   - `CUDA_ARCH=sm_86` 或 `ARCH=sm_86`：指定生成 PTX 的目标架构（默认 `sm_89`）。
   - `CUDA_RREG=64` 或 `RREG=64`：为 nvcc 添加 `-maxrregcount` 限制（可依硬件调参）。
   - `NVCC_FLAGS="--extra-options"`：追加自定义 nvcc 参数（以空格分隔）。

## 构建与运行

### 1. 仅验证代码（无 GPU 环境）
默认情况下 `gpu` crate 启用 `stub` 特性，提供占位实现，便于在没有 CUDA 的开发机上编译：
```bash
cargo build --release
```

### 2. 单独编译 GPU 模块（可选）
在有 CUDA 环境时，你可以先确定 GPU crate 能否独立编译成功：
```bash
ARCH=sm_89 RREG=64 cargo build -p gpu --release --no-default-features --features cuda
```
`ARCH`/`RREG` 可按需替换为目标架构与寄存器限制，若省略则使用默认值。

### 3. 编译并运行 Server（会同时编译 GPU 模块）
```bash
ARCH=sm_89 RREG=64 cargo build --release -p server --no-default-features --features gpu_cuda
PORT=8001 RUST_LOG=info ARCH=sm_89 RREG=64 \
  cargo run --release -p server --no-default-features --features gpu_cuda
```
说明：
- 如果未先执行 `cargo build`，`cargo run` 会自动进行一次编译。
- 只在需要调优时设置 `ARCH`、`RREG` 等环境变量；否则可以直接省略，使用默认 `sm_89` 与编译器自动分配的寄存器数。
- `PORT` 控制监听端口（默认为 `8001`），`RUST_LOG=info` 会打印基本运行日志。
- 也可以直接运行已编好的二进制：`PORT=9000 RUST_LOG=info ./target/release/server`。

启动日志示例：
```
$ PORT=9000 RUST_LOG=info cargo run --release -p server --no-default-features --features gpu_cuda
...
INFO server: listening on http://0.0.0.0:9000
```
日志中带 `http://...` 的行表示服务已可对外提供 API。

服务默认监听 `0.0.0.0:8001`，可通过环境变量 `PORT` 覆盖。API 采用“任务”模式：

- `POST /api/v1/jobs`：提交新任务。
- `GET /api/v1/jobs/{outpoint}`：查询任务状态。

### 创建任务请求

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `outpoint` | string | 必填 | 任务 ID，同样作为挖矿 challenge（形如 `txid:vout`） |
| `start_nonce` | u64 | 0 | nonce 起始值 |
| `total_nonce` | u64 | 5000 亿 | 整次请求要搜索的 nonce 总量 |
| `batch_size` | u32 | 1,000,000 | 默认单次 kernel launch 的目标 nonce 数 |
| `threads_per_block` | u32 | 256 | CUDA block 大小，用于调优 |
| `blocks` | u32 | 自动推导 | CUDA grid 中的 block 数；若指定则与 `threads_per_block`、`ilp` 决定单次 launch 覆盖量 |
| `ilp` | u32 | 1 | Instruction-level parallelism，单线程连续处理的 nonce 数 |
| `persistent` | bool | false | 若为 `true`，任务成功完成后再次提交同一 `outpoint` 会直接返回缓存结果 |

参数关系说明：
- `total_nonce` 控制整次请求要完成的搜索量，服务会按 `batch_size`（或显式 `blocks * threads_per_block * ilp`）拆分成多次 kernel launch。
- 未显式提供 `blocks` 时，服务会根据 `batch_size`、`threads_per_block` 与 `ilp` 自动推导能覆盖目标 nonce 的 block 数。
- 如果提供了 `blocks`，单次 launch 的理论覆盖量为 `blocks * threads_per_block * ilp`，并与 `batch_size` 取最小值；可用来直接套用历史调优数据。
- 调整 `threads_per_block` 与 `ilp` 可以针对不同 GPU 的 SM 结构与指令吞吐测试最佳配置。
- `persistent=true` 表示缓存任务结果：同一 `outpoint` 再次提交将立即返回之前的执行结果；`false` 则会重新创建任务。

请求示例：
```json
{
  "outpoint": "5f6a49b12a3275a3d92f05aa9c41792363f7ca71b78e2e178310e22da4ad3a9:2",
  "start_nonce": 0,
  "total_nonce": 500000000000,
  "batch_size": 2097152,
  "threads_per_block": 512,
  "blocks": 4096,
  "ilp": 8,
  "persistent": true
}
```

### 任务状态响应

无论是提交任务（202 Accepted）还是查询任务（200 OK），响应结构相同：

```json
{
  "outpoint": "...",
  "persistent": true,
  "status": "running",
  "result": {
    "outpoint": "...",
    "best_nonce": "123456",
    "best_lz": 23,
    "best_hash": "...",
    "searched": 1000000,
    "start_nonce": "0"
  },
  "error": null
}
```

当任务尚未完成时 `result` 为空；失败时 `status` 为 `failed` 并在 `error` 字段给出原因。

## 常见问题
- **`Could not find a cuda installation`**：`cust` 的构建脚本未找到 CUDA Toolkit。确认驱动与 CUDA 安装完整，并检查 `CUDA_HOME`、`PATH`、`LD_LIBRARY_PATH`。
- **`Failed to run nvcc`**：`nvcc` 不在 PATH 或缺失。设置 `NVCC` 环境变量指向实际位置，或重新安装 CUDA Toolkit。
- **不同 GPU 架构**：编译时设置 `CUDA_ARCH`（例如 `sm_75`、`sm_80`）与目标 GPU 的 Compute Capability 对应。
- **`gpu crate was built without the cuda feature`**：当前链接的是 stub 实现。按照上文“启用真实 CUDA 实现”一节使用 `--no-default-features --features gpu_cuda` 重新构建。
- **大规模 `total_nonce`**：`total_nonce` 表示整次请求要搜索的 nonce 总量，默认 5000 亿。内部会按照 `batch_size`（默认 100 万）拆分成多次 GPU 调用，必要时可根据 GPU 资源调整该值。
- **调参 `blocks` / `threads_per_block` / `ilp`**：若需要针对硬件调优，可在请求里显式指定。`batch_size` 仍然用于限制单次调用的最大 nonce 数，当 `blocks` 提供时会以 `blocks * threads_per_block * ilp` 作为每次调用的上限。

## 后续计划
- 更新 `Dockerfile` 以适配新的工作区结构。
- 根据需要引入 CPU fallback 或简化的模拟实现，方便无 GPU 环境的开发验证。
