#!/bin/bash

# RTX 4090 优化运行脚本

echo "=== RTX 4090 性能优化运行脚本 ==="
echo "优化配置："
echo "- 线程块: 256 threads/block"
echo "- 网格: 4096 blocks"
echo "- ILP: 16 (最大指令级并行)"
echo "- Chunk: 262144"
echo "- Persistent模式: 启用"
echo ""

# 设置CUDA优化环境变量
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB CUDA缓存
export CUDA_FORCE_PTX_JIT=1           # 强制PTX JIT编译
export CUDA_LAUNCH_BLOCKING=0         # 异步执行

# 编译优化的CUDA内核
echo "编译CUDA内核..."
ARCH=sm_89 RREG=128 ./build_cubin_ada.sh

# 编译Rust代码
echo "编译Rust代码..."
cargo build --release

# 运行API服务器
echo "启动优化的挖矿服务..."
cargo run --release

# 提示如何测试
echo ""
echo "=== 测试命令 ==="
echo "使用以下curl命令测试性能："
echo ""
cat << 'EOF'
curl -X POST http://localhost:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "outpoint": "8a4b24e948315a338ad421a1d01e14260b7e697291f1fb0c44e64829a7fa80cd:1",
    "wait": true,
    "options": {
      "total_nonce": 100000000000,
      "threads_per_block": 256,
      "blocks": 4096,
      "ilp": 16,
      "persistent": true,
      "chunk_size": 262144,
      "batch_size": 50000000000,
      "progress_ms": 1000,
      "binary_nonce": false,
      "odometer": true
    }
  }'
EOF