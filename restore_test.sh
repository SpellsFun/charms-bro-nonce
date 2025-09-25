#!/bin/bash

# 恢复测试 - 使用之前达到7.7 GH/s的配置

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 恢复到原始配置测试 ===${NC}\n"

# 重新编译
echo -e "${YELLOW}重新编译...${NC}"
ARCH=sm_89 ./build_cubin_ada.sh
cargo build --release

# 重启服务
echo -e "${YELLOW}重启服务...${NC}"
pkill -f "target/release/bro" || true
sleep 2
nohup cargo run --release > server.log 2>&1 &
sleep 3

# 测试原始成功的配置
echo -e "\n${YELLOW}测试原始配置（之前达到7.7 GH/s）${NC}"

OUTPOINT="restore_test_$(date +%s):1"

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
      \"progress_ms\": 1000
    }
  }" | python3 -m json.tool

echo -e "\n${GREEN}如果性能仍然低于7.7 GH/s，问题可能是：${NC}"
echo "1. 系统环境变化（GPU温度、频率）"
echo "2. 其他进程占用GPU"
echo "3. 驱动或CUDA版本变化"
echo ""
echo "检查GPU状态："
echo "  nvidia-smi"
echo "  nvidia-smi -q -d PERFORMANCE"