#!/bin/bash

# RTX 4090 最优配置测试

echo "=== RTX 4090 优化测试 ==="
echo ""
echo "注意: 使用ASCII模式（binary_nonce=false）以确保结果正确"
echo ""

# 生成唯一outpoint
RAND=$RANDOM$RANDOM
OUTPOINT="test${RAND}:1"

# 最优配置测试
echo "测试配置:"
echo "- 线程: 256 threads/block"
echo "- 块数: 2048 blocks"
echo "- ILP: 16"
echo "- Chunk: 262144"
echo "- ASCII模式"
echo ""

curl -X POST http://localhost:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d "{
    \"outpoint\": \"${OUTPOINT}\",
    \"wait\": true,
    \"options\": {
      \"total_nonce\": 100000000000,
      \"threads_per_block\": 256,
      \"blocks\": 2048,
      \"ilp\": 16,
      \"persistent\": true,
      \"chunk_size\": 262144,
      \"binary_nonce\": false,
      \"odometer\": true,
      \"progress_ms\": 1000
    }
  }" | python3 -m json.tool

echo ""
echo "如果性能仍低于10 GH/s，尝试以下配置:"
echo ""
echo "1. 更多块:"
echo "   blocks: 4096"
echo ""
echo "2. 更大chunk:"
echo "   chunk_size: 524288"
echo ""
echo "3. 更高ILP:"
echo "   ilp: 32"