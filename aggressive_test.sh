#!/bin/bash

# 激进优化测试 - 尝试达到更高性能

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== RTX 4090 激进优化测试 ===${NC}\n"

# 测试函数
test_config() {
    local name="$1"
    local config="$2"

    local outpoint="aggressive$(date +%s%N):1"

    echo -e "${YELLOW}测试: $name${NC}"
    echo "配置: $config"

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
        echo -e "  性能: ${GREEN}${rate} GH/s${NC}"
        echo -e "  耗时: ${duration}秒\n"
        echo "$name|$rate" >> aggressive_results.txt
    else
        echo -e "  ${RED}失败${NC}\n"
    fi

    sleep 1
}

# 清空结果
> aggressive_results.txt

echo "注意: 使用更大的工作量以获得准确结果"
echo ""

# 测试1: 之前表现最好的配置（7.7 GH/s）
test_config "原始最优-256x2048xILP16" '{
    "total_nonce": 200000000000,
    "threads_per_block": 256,
    "blocks": 2048,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 262144,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 10000000000,
    "progress_ms": 2000
}'

# 测试2: 基于当前最佳的512x1024
test_config "当前最优-512x1024xILP16" '{
    "total_nonce": 200000000000,
    "threads_per_block": 512,
    "blocks": 1024,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 262144,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 10000000000,
    "progress_ms": 2000
}'

# 测试3: 超大块数
test_config "超大块-256x8192xILP8" '{
    "total_nonce": 200000000000,
    "threads_per_block": 256,
    "blocks": 8192,
    "ilp": 8,
    "persistent": true,
    "chunk_size": 131072,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 20000000000,
    "progress_ms": 2000
}'

# 测试4: 平衡配置
test_config "平衡-384x1536xILP12" '{
    "total_nonce": 200000000000,
    "threads_per_block": 384,
    "blocks": 1536,
    "ilp": 12,
    "persistent": true,
    "chunk_size": 196608,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 15000000000,
    "progress_ms": 2000
}'

# 测试5: 小线程大ILP
test_config "小线程大ILP-128x2048xILP32" '{
    "total_nonce": 200000000000,
    "threads_per_block": 128,
    "blocks": 2048,
    "ilp": 32,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 10000000000,
    "progress_ms": 2000
}'

# 测试6: 大chunk配置
test_config "大chunk-256x2048xILP16xC1M" '{
    "total_nonce": 200000000000,
    "threads_per_block": 256,
    "blocks": 2048,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 1048576,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 50000000000,
    "progress_ms": 2000
}'

# 测试7: 非persistent模式对比
test_config "非持久-256x2048xILP16" '{
    "total_nonce": 200000000000,
    "threads_per_block": 256,
    "blocks": 2048,
    "ilp": 16,
    "persistent": false,
    "chunk_size": 262144,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 10000000000,
    "progress_ms": 2000
}'

# 测试8: 关闭odometer
test_config "无odometer-256x2048xILP16" '{
    "total_nonce": 200000000000,
    "threads_per_block": 256,
    "blocks": 2048,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 262144,
    "binary_nonce": false,
    "odometer": false,
    "batch_size": 10000000000,
    "progress_ms": 2000
}'

echo -e "${GREEN}=== 测试完成 ===${NC}\n"

# 显示结果
echo "结果汇总:"
echo "配置 | GH/s"
echo "-----|-----"
cat aggressive_results.txt | column -t -s'|'

echo ""
echo -e "${GREEN}最佳配置:${NC}"
sort -t'|' -k2 -rn aggressive_results.txt | head -1 | while IFS='|' read name rate; do
    echo "  $name: ${rate} GH/s"
done

# 分析
echo -e "\n${YELLOW}分析:${NC}"
echo "1. 如果所有结果都在6-7 GH/s，可能是："
echo "   - SHA256双哈希的理论限制"
echo "   - 内存带宽瓶颈"
echo "   - 需要更底层的优化"
echo ""
echo "2. 如果某个配置达到8+ GH/s，记录该配置"
echo ""
echo "3. 检查GPU状态:"
echo "   nvidia-smi -q -d PERFORMANCE"
echo "   nvidia-smi dmon -i 0 -s pucvmet"