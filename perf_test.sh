#!/bin/bash

# 直接性能测试脚本 - 绕过缓存问题

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== RTX 4090 性能测试 ===${NC}\n"

# 确保服务运行
if ! curl -s "http://localhost:8001/api/v1/jobs" > /dev/null 2>&1; then
    echo -e "${YELLOW}启动服务...${NC}"
    cargo build --release
    nohup cargo run --release > server.log 2>&1 &
    sleep 3
fi

# 测试函数
test_config() {
    local name="$1"
    local threads="$2"
    local blocks="$3"
    local ilp="$4"
    local chunk="$5"
    local binary="$6"

    # 生成唯一的outpoint
    local rand=$RANDOM$RANDOM
    local outpoint="test${rand}:1"

    echo -e "${YELLOW}测试: $name${NC}"
    echo "  配置: threads=$threads blocks=$blocks ilp=$ilp chunk=$chunk binary=$binary"
    echo -n "  运行... "

    local start=$(date +%s.%N)

    local response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"$outpoint\",
            \"wait\": true,
            \"options\": {
                \"total_nonce\": 10000000000,
                \"threads_per_block\": $threads,
                \"blocks\": $blocks,
                \"ilp\": $ilp,
                \"persistent\": true,
                \"chunk_size\": $chunk,
                \"binary_nonce\": $binary,
                \"progress_ms\": 0
            }
        }")

    local end=$(date +%s.%N)
    local duration=$(echo "$end - $start" | bc)

    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo -e "${GREEN}完成${NC}"
        echo "  性能: ${rate} GH/s"
        echo "  耗时: ${duration}秒"
        echo "$name|$rate|$duration" >> perf_results.txt
    else
        echo -e "${RED}失败${NC}"
        echo "  $response"
    fi

    # 短暂延迟避免过载
    sleep 1
}

# 清空结果文件
echo "Test|GH/s|Duration" > perf_results.txt

# 运行测试组合
echo -e "\n${GREEN}开始性能测试...${NC}\n"

# 基准测试
test_config "基准-256x2048" 256 2048 16 262144 false

# 不同线程配置
test_config "线程-512x1024" 512 1024 16 262144 false
test_config "线程-128x4096" 128 4096 16 262144 false

# 不同ILP
test_config "ILP-4" 256 2048 4 262144 false
test_config "ILP-8" 256 2048 8 262144 false
test_config "ILP-32" 256 2048 32 262144 false

# 不同chunk大小
test_config "Chunk-131K" 256 2048 16 131072 false
test_config "Chunk-524K" 256 2048 16 524288 false

# 不同块数
test_config "块-1024" 256 1024 16 262144 false
test_config "块-4096" 256 4096 16 262144 false

# 极限测试
test_config "极限-256x4096x32" 256 4096 32 524288 false
test_config "极限-128x8192x16" 128 8192 16 262144 false

echo -e "\n${GREEN}=== 结果汇总 ===${NC}"
column -t -s'|' perf_results.txt

echo -e "\n${GREEN}最佳配置:${NC}"
tail -n +2 perf_results.txt | sort -t'|' -k2 -rn | head -1 | while IFS='|' read name rate duration; do
    echo "  $name: ${rate} GH/s (${duration}秒)"
done