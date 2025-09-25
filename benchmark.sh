#!/bin/bash

# RTX 4090 自动化性能测试脚本
# 用于测试和优化GPU挖矿性能

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
SERVER_URL="http://localhost:8001"
OUTPOINT="8a4b24e948315a338ad421a1d01e14260b7e697291f1fb0c44e64829a7fa80cd:1"
TOTAL_NONCE=10000000000  # 10B for quick test

echo -e "${GREEN}=== RTX 4090 性能基准测试 ===${NC}"
echo ""

# 检查服务是否运行
check_server() {
    echo -n "检查服务器状态... "
    if curl -s "${SERVER_URL}/api/v1/jobs" > /dev/null 2>&1; then
        echo -e "${GREEN}运行中${NC}"
        return 0
    else
        echo -e "${RED}未运行${NC}"
        echo "请先运行: cargo run --release"
        return 1
    fi
}

# 运行测试
run_test() {
    local name=$1
    local config=$2

    echo -e "\n${YELLOW}测试: $name${NC}"
    echo "配置: $config"
    echo -n "运行中... "

    local start_time=$(date +%s)

    local response=$(curl -s -X POST "${SERVER_URL}/api/v1/jobs" \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"${OUTPOINT}\",
            \"wait\": true,
            \"options\": ${config}
        }")

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # 解析结果
    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)
    local best_lz=$(echo "$response" | grep -o '"best_lz":[0-9]*' | cut -d: -f2)
    local best_nonce=$(echo "$response" | grep -o '"best_nonce":[0-9]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo -e "${GREEN}完成${NC}"
        echo "  性能: ${rate} GH/s"
        echo "  最佳LZ: ${best_lz}"
        echo "  最佳Nonce: ${best_nonce}"
        echo "  耗时: ${duration}秒"
        echo "$name,$rate,$best_lz,$duration" >> benchmark_results.csv
    else
        echo -e "${RED}失败${NC}"
        echo "  响应: $response"
    fi

    return 0
}

# 编译优化内核
compile_kernel() {
    echo -e "${YELLOW}编译CUDA内核...${NC}"
    if [ -f "build_cubin_ada.sh" ]; then
        ARCH=sm_89 RREG=128 ./build_cubin_ada.sh > /dev/null 2>&1
        echo -e "${GREEN}编译完成${NC}"
    else
        echo -e "${YELLOW}跳过编译（build脚本不存在）${NC}"
    fi
}

# 主测试流程
main() {
    # 检查服务器
    if ! check_server; then
        exit 1
    fi

    # 初始化结果文件
    echo "Test,GH/s,Best_LZ,Duration" > benchmark_results.csv

    # 测试1: 默认配置
    run_test "默认配置" '{
        "total_nonce": 10000000000,
        "persistent": true,
        "progress_ms": 1000
    }'

    # 测试2: 优化配置 - 二进制模式
    run_test "优化-二进制" '{
        "total_nonce": 10000000000,
        "threads_per_block": 256,
        "blocks": 4096,
        "ilp": 16,
        "persistent": true,
        "chunk_size": 262144,
        "binary_nonce": true,
        "progress_ms": 1000
    }'

    # 测试3: 优化配置 - ASCII模式
    run_test "优化-ASCII" '{
        "total_nonce": 10000000000,
        "threads_per_block": 256,
        "blocks": 4096,
        "ilp": 16,
        "persistent": true,
        "chunk_size": 262144,
        "binary_nonce": false,
        "odometer": true,
        "progress_ms": 1000
    }'

    # 测试4: 极限配置
    run_test "极限配置" '{
        "total_nonce": 10000000000,
        "threads_per_block": 128,
        "blocks": 8192,
        "ilp": 32,
        "persistent": true,
        "chunk_size": 524288,
        "binary_nonce": true,
        "progress_ms": 1000
    }'

    # 显示结果
    echo -e "\n${GREEN}=== 测试结果汇总 ===${NC}"
    column -t -s',' benchmark_results.csv

    # 找出最佳配置
    echo -e "\n${GREEN}最佳配置:${NC}"
    tail -n +2 benchmark_results.csv | sort -t',' -k2 -rn | head -1 | while IFS=',' read name rate lz duration; do
        echo "  $name: ${rate} GH/s"
    done

    echo -e "\n${GREEN}测试完成！${NC}"
    echo "详细结果保存在: benchmark_results.csv"
}

# 如果提供了参数，运行特定测试
if [ "$1" == "compile" ]; then
    compile_kernel
    exit 0
elif [ "$1" == "quick" ]; then
    # 快速测试
    check_server && run_test "快速测试" '{
        "total_nonce": 1000000000,
        "threads_per_block": 256,
        "blocks": 4096,
        "ilp": 16,
        "persistent": true,
        "chunk_size": 262144,
        "binary_nonce": true
    }'
    exit 0
elif [ "$1" == "custom" ]; then
    # 自定义测试
    if [ -z "$2" ]; then
        echo "用法: $0 custom '{\"total_nonce\": 1000000000, ...}'"
        exit 1
    fi
    check_server && run_test "自定义测试" "$2"
    exit 0
else
    main
fi