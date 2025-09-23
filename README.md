# bro: 多 GPU 双重 SHA‑256 碰撞搜索器（Rust + CUDA）

本项目在 GPU 上并行搜索使 `SHA256(SHA256(outpoint + nonce))` 具有最多前导零位（leading zeros）的 `nonce`，并提供 REST API 供外部调用。

- 多 GPU 并行：按权重切分工作量，每张卡独立推进进度。
- 两种执行模式：
  - 持久化内核（默认）：设备端自取任务，减少往返开销，适合长时间跑满。
  - 批处理模式：显式关闭持久化，主机按批次下发工作，便于调试。
- `nonce` 默认使用 8 字节二进制，也可切换回十进制 ASCII 以兼容旧流程。
- 内核加载：优先加载 `sha256_kernel.cubin`，失败时回退 `sha256_kernel.ptx`（均由 `nvcc` 生成）。
- REST API：通过 Axum + Tokio 提供 `/api/v1/jobs` 任务接口，外部系统可提交搜索任务、查询状态与结果，取代旧版的命令行交互模式。


**环境要求**
- NVIDIA GPU 与驱动（可用 `nvidia-smi` 检查）。
- CUDA Toolkit（含 `nvcc`，用于构建 CUBIN/PTX）。
- Rust 工具链（`cargo`，edition 2021）。
- 建议 Linux 环境；Windows/WSL 需确保 CUDA 可用。macOS 无原生 NVIDIA 支持。


**快速安装与启动流程**
- 安装 Rust（按提示完成安装）：
  - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- 配置环境变量（当前会话生效，建议写入 shell 配置文件）：
  - `export PATH="$HOME/.cargo/bin:$PATH"`
- 检查是否安装成功：
  - `rustc --version`
  - `cargo --version`
- 赋予构建脚本执行权限（首次或上传后执行一次）：
  - `chmod +x build_cubin_ada.sh`
- 生成 CUBIN（失败时自动回退 PTX，通常只需执行一次）：
  - `./build_cubin_ada.sh`
- 启动 REST API 服务（替换旧版 CLI/交互模式）：
  - `PORT=3000 cargo run --release`
- 查看 GPU 占用率：
  - `nvidia-smi`

注：以上“生成 CUBIN/赋权”通常只需运行一次；后续根据需要直接“启动”。


**目录结构**
- `Cargo.toml`：crate 配置，含 REST API 依赖。
- `src/lib.rs`：核心搜索逻辑与 `run_search`/`SearchConfig` API。
- `src/main.rs`：Axum 入口，暴露 `/api/v1` HTTP 接口。
- `sha256_kernel.cu`：CUDA 内核（`double_sha256_max_kernel`、`double_sha256_persistent_kernel(_ascii)`、`reduce_best_kernel` 等）。
- `build_cubin_ada.sh`：生成多架构 `sha256_kernel.cubin`，失败时回退生成 `sha256_kernel.ptx`。


**构建与运行**
- 生成内核（推荐先生成跨卡 CUBIN）：
  - 自动枚举本机架构并生成：
    - `./build_cubin_ada.sh`
  - 指定单架构：
    - `ARCH=sm_89 ./build_cubin_ada.sh`
  - 指定多架构：
    - `ARCHES="sm_89,sm_120" ./build_cubin_ada.sh`
  - 限制寄存器（可影响性能/占用）：
    - `RREG=64 ./build_cubin_ada.sh`
- 启动 REST API 服务（默认端口 8001，可通过 `PORT` 覆盖）：
  - `cargo run --release`
- 停止服务：`Ctrl+C`；持久化模式下会等待当前 chunk 收尾。


**Docker 镜像**
- 需要在宿主机安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，运行容器时通过 `--gpus` 暴露 GPU。
- 构建镜像（默认编译 `sm_89` 架构，可通过 `--build-arg CUDA_ARCHES=sm_89,sm_75` 自定义多个目标）：
  - `docker build -t charms-bro-nonce .`
- 运行：
  - `docker run --rm --gpus all -p 8001:8001 charms-bro-nonce`
- 可覆写端口和环境，例如：
  - `docker run --rm --gpus all -e PORT=8080 -p 8080:8080 charms-bro-nonce`
- 镜像工作目录 `/app` 下已预置 `sha256_kernel.cubin/ptx`；需要重新生成时可在构建阶段调整 `CUDA_ARCHES` 或修改 `build_cubin_ada.sh`。
- 推送到 Docker Hub（假设仓库为 `youruser/charms-bro-nonce`）：
  - `docker tag charms-bro-nonce youruser/charms-bro-nonce:latest`
  - `docker login`
  - `docker push youruser/charms-bro-nonce:latest`


**API 使用**
- 创建任务：`POST /api/v1/jobs`
  ```bash
  curl -X POST http://localhost:8001/api/v1/jobs \
    -H 'content-type: application/json' \
    -d '{
          "outpoint": "txid:index",
          "options": {
            "total_nonce": 100000000000
          },
          "wait": false
        }'
  ```
- 默认行为：立即返回 `202 Accepted` 和 `job_id`，任务在后台继续执行。
- 设 `wait=true` 时请求会同步阻塞到任务完成，响应中附带最终结果；使用该模式时请确保客户端超时设置足够大。
- 再次提交相同 `outpoint` 会直接返回已有任务状态，不会并行启动重复搜索；如需重新搜索可在任务消失后重新提交。
- 查询单个任务：`GET /api/v1/jobs/{job_id}`，字段含 `status`（pending/running/completed/failed）、时间戳、结果或错误信息；若任务不存在（例如服务重启后），会自动以默认参数重启同名任务并返回 `202 Accepted` 的 `pending` 状态。
- 列出全部任务：`GET /api/v1/jobs`，按提交时间返回所有任务快照，可用于轮询进度。


**任务参数（POST /api/v1/jobs 的 options）**
- `total_nonce` (`u64`，默认 `100_000_000_000_000`)：最大搜索次数上限，支持十进制整数。
- `start_nonce` (`u64`，默认 `0`)：搜索起始 nonce。
- `batch_size` (`u64`，默认 `1_000_000_000`)：批处理模式下一次下发的工作量；`0` 表示单批完成。
- `threads_per_block` (`u32`，默认 `512`)：每个 block 的线程数。
- `blocks` (`u32`，默认 `4096`)：一次 launch 的 block 数。
- `binary_nonce` (`bool`，默认 `true`)：`true` 使用 8 字节二进制 nonce，`false` 走十进制 ASCII。
- `persistent` (`bool`，默认 `true`)：是否启用持久化内核。
- `chunk_size` (`u32`，默认 `2097152`)：持久化模式每次抓取的 chunk 大小。
- `ilp` (`u32`，默认 `8`)：内核内指令级并行度。
- `progress_ms` (`u64`，默认 `0`)：>0 时服务端周期性打印进度到 stdout。
- `odometer` (`bool`，默认 `true`)：持久化内核是否启用计数器输出；当 `binary_nonce=true` 时会自动关闭。
- `gpu_ids` (`[u32]`，默认全卡)：指定使用的 GPU 索引。
- `gpu_weights` (`[f64]`)：与 `gpu_ids` 对齐的权重，决定工作量占比。

根级字段：
- `outpoint` (`string`，必填)：通常为 `txid:index` 这样的交易输出标识，会同时作为搜索前缀与 `job_id`。
- `wait` (`bool`，默认 `false`)：`true` 时同步等待任务完成，`false`/省略时立即返回 `job_id`。

`outpoint` 字符串需满足 `len(outpoint) + (ASCII 时 20 | 二进制时 8) <= 128`，否则任务会立刻失败并返回错误信息。


**与旧版 CLI 的区别**
- 入口改为 REST 服务，所有参数通过 JSON `options` 传递，不再提示命令行输入或需要交互式确认。
- 后端仍按 `job_id` 排队串行执行（默认单 worker），客户端可选择立即得到 `job_id` 再查询，或以 `wait=true` 等待同步结果。
- 若仍需脚本化调用，可在外部封装 HTTP 请求；无需再直接运行 `cargo run --release -- "outpoint"` 之类命令。


**持久化模式与性能调优**
- 持久化（`persistent=true`）会在 GPU 上长驻线程块，自行从全局计数器领取工作；适合长时间高负载任务。
- RTX 4090 / Ada 架构实测：默认组合 `threads_per_block=512`、`blocks=4096`、`chunk_size=2097152`、`ilp=8`、`binary_nonce=true`、`RREG=64` 可达到约 15–16 GH/s，可在此基础上微调。
- 若需要降低延迟，可把 `chunk_size` 调小（例如 `524288`）；若想进一步降低 `atomicAdd` 压力，可适度增大，但要注意进度反馈会更慢。
- 切换回 ASCII nonce 时，请显式设置 `binary_nonce=false`；服务端会自动重新启用 odometer 以保持十进制自增。
- 多 GPU 时，可通过 `gpu_weights` 在单次任务内按性能分配工作量，或在服务层外部复制多个请求按需拆分。


**常见问题**
- 报错 `kernel module not found`：先运行 `./build_cubin_ada.sh` 生成 `sha256_kernel.cubin` 或 `sha256_kernel.ptx`。
- 找不到 `nvcc`：请安装 CUDA Toolkit 并将 `nvcc` 加入 PATH。
- 没有 `nvidia-smi`：无法自动检测架构，请用 `ARCH/ARCHES` 明确指定目标架构。
- 输入过长：需满足 `len(outpoint) + 20(ASCII) 或 8(二进制) <= 128`。
- 性能异常：优先使用 CUBIN；根据上述“手动性能调优”逐项验证 `THREADS/BLOCKS/ILP/CHUNK/RREG`。
- 中断退出：按 Ctrl+C；持久化模式下会触发设备端停止标志，稍候清理完成。


**注意事项**
- 长时间运行会高负载占用 GPU，请关注温度与功耗（可能需要设置功耗/频率上限）。
- 本项目仅供研究与学习用途，请在合法合规前提下使用。
