#!/bin/bash

# RTX 4090 极限优化脚本

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== RTX 4090 极限优化 ===${NC}\n"

# 1. GPU状态检查
echo -e "${YELLOW}步骤1: GPU状态检查${NC}"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics --format=csv
echo ""

# 2. 编译多个优化版本
echo -e "${YELLOW}步骤2: 编译多个优化版本${NC}"

# 版本1: 标准优化
echo "编译版本1: 标准优化..."
nvcc -O3 -arch=sm_89 -maxrregcount=64 -Xptxas -O3,-v -Xptxas -dlcm=ca -cubin sha256_kernel.cu -o sha256_v1.cubin 2>/dev/null || echo "v1 failed"

# 版本2: 高寄存器
echo "编译版本2: 高寄存器..."
nvcc -O3 -arch=sm_89 -maxrregcount=128 -Xptxas -O3,-v -Xptxas -dlcm=cg -use_fast_math -cubin sha256_kernel.cu -o sha256_v2.cubin 2>/dev/null || echo "v2 failed"

# 版本3: 激进优化
echo "编译版本3: 激进优化..."
nvcc -O3 -arch=sm_89 -maxrregcount=96 \
  -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false \
  -Xptxas -O3,-v -Xptxas -dlcm=ca \
  -Xcompiler -O3,-march=native,-mtune=native,-ffast-math \
  -cubin sha256_kernel.cu -o sha256_v3.cubin 2>/dev/null || echo "v3 failed"

# 使用最成功的版本
if [ -f sha256_v3.cubin ]; then
    cp sha256_v3.cubin sha256_kernel.cubin
    echo -e "${GREEN}使用激进优化版本${NC}"
elif [ -f sha256_v2.cubin ]; then
    cp sha256_v2.cubin sha256_kernel.cubin
    echo -e "${YELLOW}使用高寄存器版本${NC}"
elif [ -f sha256_v1.cubin ]; then
    cp sha256_v1.cubin sha256_kernel.cubin
    echo -e "${YELLOW}使用标准优化版本${NC}"
fi

# 3. 修改Rust默认参数以匹配最佳配置
echo -e "\n${YELLOW}步骤3: 更新默认参数${NC}"
sed -i.bak 's/pub const DEFAULT_THREADS_PER_BLOCK: u32 = .*/pub const DEFAULT_THREADS_PER_BLOCK: u32 = 128;/' src/lib.rs
sed -i.bak 's/pub const DEFAULT_BLOCKS: u32 = .*/pub const DEFAULT_BLOCKS: u32 = 1024;/' src/lib.rs
sed -i.bak 's/pub const DEFAULT_ILP: u32 = .*/pub const DEFAULT_ILP: u32 = 16;/' src/lib.rs
sed -i.bak 's/pub const DEFAULT_CHUNK_SIZE: u32 = .*/pub const DEFAULT_CHUNK_SIZE: u32 = 524288;/' src/lib.rs

# 4. 重新编译和启动
echo -e "\n${YELLOW}步骤4: 重新编译和启动服务${NC}"
cargo build --release
pkill -f "target/release/bro" || true
sleep 2
nohup cargo run --release > server.log 2>&1 &
sleep 3

# 5. 测试函数
test_extreme() {
    local name="$1"
    local config="$2"

    local outpoint="extreme_$(date +%s%N):1"

    echo -ne "${BLUE}$name: ${NC}"

    local start=$(date +%s.%N)

    local response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"$outpoint\",
            \"wait\": true,
            \"options\": $config
        }")

    local end=$(date +%s.%N)
    local duration=$(echo "$end - $start" | bc)

    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo -e "${GREEN}${rate} GH/s${NC} (${duration}s)"
        echo "$name|$rate|$duration" >> extreme_results.txt
    else
        echo -e "${RED}失败${NC}"
    fi
}

# 6. 极限测试
echo -e "\n${YELLOW}步骤5: 极限性能测试${NC}\n"

> extreme_results.txt

# 基于成功的128线程配置
echo -e "${BLUE}--- 优化的128线程配置 ---${NC}"
test_extreme "128x1024x16" '{
    "total_nonce": 100000000000,
    "threads_per_block": 128,
    "blocks": 1024,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 200000000000
}'

test_extreme "128x2048x16" '{
    "total_nonce": 100000000000,
    "threads_per_block": 128,
    "blocks": 2048,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 200000000000
}'

test_extreme "128x1024x32" '{
    "total_nonce": 100000000000,
    "threads_per_block": 128,
    "blocks": 1024,
    "ilp": 32,
    "persistent": true,
    "chunk_size": 1048576,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 200000000000
}'

# 尝试64线程以增加占用率
echo -e "\n${BLUE}--- 64线程高占用率 ---${NC}"
test_extreme "64x2048x32" '{
    "total_nonce": 100000000000,
    "threads_per_block": 64,
    "blocks": 2048,
    "ilp": 32,
    "persistent": true,
    "chunk_size": 1048576,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 200000000000
}'

test_extreme "64x4096x16" '{
    "total_nonce": 100000000000,
    "threads_per_block": 64,
    "blocks": 4096,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 200000000000
}'

# 测试非persistent模式
echo -e "\n${BLUE}--- 非Persistent模式 ---${NC}"
test_extreme "128x1024x16_NP" '{
    "total_nonce": 100000000000,
    "threads_per_block": 128,
    "blocks": 1024,
    "ilp": 16,
    "persistent": false,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 10000000000
}'

# 关闭odometer测试
echo -e "\n${BLUE}--- 无Odometer ---${NC}"
test_extreme "128x1024x16_NO" '{
    "total_nonce": 100000000000,
    "threads_per_block": 128,
    "blocks": 1024,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": false,
    "batch_size": 200000000000
}'

# 7. 结果分析
echo -e "\n${GREEN}=== 结果分析 ===${NC}\n"

echo "最佳配置："
sort -t'|' -k2 -rn extreme_results.txt | head -1 | while IFS='|' read name rate duration; do
    echo "  $name: ${rate} GH/s (耗时 ${duration}s)"
done

# 8. 性能诊断
echo -e "\n${YELLOW}性能诊断:${NC}"

best_rate=$(sort -t'|' -k2 -rn extreme_results.txt | head -1 | cut -d'|' -f2)

if (( $(echo "$best_rate > 10" | bc -l) )); then
    echo -e "${GREEN}✓ 成功突破10 GH/s！${NC}"
elif (( $(echo "$best_rate > 8" | bc -l) )); then
    echo -e "${YELLOW}接近目标，已达8+ GH/s${NC}"
    echo "建议："
    echo "1. 增加GPU功率限制: sudo nvidia-smi -pl 500"
    echo "2. 确保GPU温度<60°C"
    echo "3. 尝试更大的batch_size"
else
    echo -e "${RED}性能未达预期${NC}"
    echo "可能的限制："
    echo "1. 内存带宽瓶颈"
    echo "2. SHA256算法本身的串行依赖"
    echo "3. 需要更底层的PTX/SASS优化"
fi

# 9. 监控GPU使用率
echo -e "\n${YELLOW}GPU使用情况:${NC}"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv

echo -e "\n${BLUE}深度优化建议:${NC}"
echo "1. 如果GPU利用率<90%: 增加blocks数量"
echo "2. 如果内存利用率>80%: 减少chunk_size"
echo "3. 如果功率<400W: 提高功率限制"
echo "4. 考虑手写PTX汇编优化关键循环"