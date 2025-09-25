# RTX 5090 编译和运行指南

## 架构支持

- RTX 4090: sm_89 (Ada Lovelace)
- RTX 5090: sm_100 (Blackwell，预估)
- H100: sm_90 (Hopper)

## 本地编译

### 自动检测架构（推荐）
```bash
# 自动检测所有 GPU 架构并编译
./build_cubin_ada.sh

# 编译完成后运行
cargo run --release
```

### 手动指定架构
```bash
# 只为 5090 编译
ARCH=sm_100 ./build_cubin_ada.sh

# 同时支持 4090 和 5090
ARCHES="sm_89,sm_100" ./build_cubin_ada.sh

# 生成通用 PTX（性能略低但兼容性最好）
nvcc -O3 -ptx -arch=compute_90 sha256_kernel.cu -o sha256_kernel.ptx
```

## Docker 构建

### 自动检测（推荐）
```bash
# 自动检测构建机器的 GPU 架构
docker build -t charms-bro-nonce:auto .
```

### 指定架构
```bash
# 为 5090 构建
docker build --build-arg CUDA_ARCHES="sm_100" -t charms-bro-nonce:5090 .

# 多架构支持
docker build --build-arg CUDA_ARCHES="sm_89,sm_90,sm_100" -t charms-bro-nonce:multi .
```

### 使用更新的 CUDA 版本（5090 可能需要）
```bash
docker build --build-arg CUDA_VERSION=12.6.0 -t charms-bro-nonce:cuda126 .
```

## 运行

### 基本运行
```bash
# Docker
docker run --rm --gpus all -p 8001:8001 charms-bro-nonce:auto

# 本地
cargo run --release
```

### API 调用 - 5090 优化参数
```bash
curl -X POST http://localhost:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "outpoint": "your_outpoint_here",
    "options": {
      "total_nonce": 100000000000,
      "start_nonce": 0,
      "threads_per_block": 768,    # 5090 建议值
      "blocks": 12288,              # 5090 建议值
      "ilp": 8,                     # 5090 可以支持更高的 ILP
      "chunk_size": 4194304,        # 增大 chunk size
      "persistent": true,
      "progress_ms": 1000
    }
  }'
```

## 性能调优

### RTX 5090 建议配置

#### 最大性能模式
```json
{
  "threads_per_block": 1024,
  "blocks": 16384,
  "ilp": 8,
  "chunk_size": 8388608
}
```

#### 平衡模式（推荐）
```json
{
  "threads_per_block": 768,
  "blocks": 12288,
  "ilp": 6,
  "chunk_size": 4194304
}
```

#### 节能模式
```json
{
  "threads_per_block": 512,
  "blocks": 8192,
  "ilp": 4,
  "chunk_size": 2097152
}
```

## 故障排除

### 1. CUDA 版本不兼容
```bash
# 错误：cuda>=12.6, please update your driver
# 解决：升级驱动或使用较低版本的 CUDA 镜像
docker build --build-arg CUDA_VERSION=12.5.1 -t charms-bro-nonce .
```

### 2. 架构不支持
```bash
# 错误：no kernel image is available for execution
# 解决：使用 PTX 文件
nvcc -O3 -ptx -arch=compute_90 sha256_kernel.cu -o sha256_kernel.ptx
```

### 3. 性能调试
```bash
# 监控 GPU 使用率
nvidia-smi dmon -s pucvmet -i 0

# 查看详细信息
nvidia-smi -q -d PERFORMANCE
```

## 预期性能

| GPU | 架构 | CUDA 核心 | 预期算力 (GH/s) |
|-----|------|----------|----------------|
| RTX 4090 | sm_89 | 16,384 | 7-8 |
| RTX 5090 | sm_100 | ~28,000 | 12-14 |

## 注意事项

1. **驱动要求**：RTX 5090 需要 NVIDIA 驱动 560+
2. **CUDA 版本**：建议使用 CUDA 12.6 或更新版本
3. **功耗**：5090 功耗可能达到 500-600W，确保电源充足
4. **温度**：监控 GPU 温度，必要时调整参数避免过热

## 架构检测

程序运行时会显示 GPU 架构信息：
```
Launching 1 GPU worker(s): [0]
  - GPU 0: NVIDIA GeForce RTX 5090 (sm_100)
```

如果显示的架构与预期不符，请检查驱动和 CUDA 版本。