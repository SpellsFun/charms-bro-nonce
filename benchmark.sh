#!/bin/bash

# RTX 4090 自动化性能测试脚本
# 编译、启动服务并运行性能测试

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

# 环境变量
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_FORCE_PTX_JIT=1

echo -e "${GREEN}=== RTX 4090 性能基准测试 ===${NC}"
echo ""

# 编译项目
compile_project() {
    echo -e "${YELLOW}编译项目...${NC}"

    # 编译CUDA内核
    if [ -f "build_cubin_ada.sh" ]; then
        echo "  编译CUDA内核..."
        ARCH=sm_89 RREG=128 ./build_cubin_ada.sh || true
    fi

    # 编译Rust
    echo "  编译Rust代码..."
    cargo build --release

    echo -e "${GREEN}编译完成${NC}"
}

# 启动服务
start_server() {
    echo -e "${YELLOW}启动服务...${NC}"

    # 停止旧服务
    pkill -f "target/release/bro" || true
    sleep 1

    # 后台启动服务
    nohup cargo run --release > server.log 2>&1 &
    local pid=$!

    # 等待服务启动
    echo -n "等待服务启动"
    for i in {1..10}; do
        sleep 1
        echo -n "."
        if curl -s "${SERVER_URL}/api/v1/jobs" > /dev/null 2>&1; then
            echo -e " ${GREEN}成功${NC}"
            echo "服务PID: $pid"
            return 0
        fi
    done

    echo -e " ${RED}失败${NC}"
    echo "服务日志:"
    tail -n 20 server.log
    return 1
}

# 检查服务是否运行
check_server() {
    if curl -s "${SERVER_URL}/api/v1/jobs" > /dev/null 2>&1; then
        return 0
    else
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

# 停止服务
stop_server() {
    echo -e "${YELLOW}停止服务...${NC}"
    pkill -f "target/release/bro" || true
    sleep 1
    echo -e "${GREEN}服务已停止${NC}"
}

# 主测试流程
main() {
    # 根据参数执行不同操作
    case "${1:-test}" in
        compile)
            compile_project
            ;;
        start)
            compile_project
            start_server
            ;;
        stop)
            stop_server
            ;;
        test)
            # 检查服务是否运行，如果没有则编译并启动
            if ! check_server; then
                echo -e "${YELLOW}服务未运行，正在启动...${NC}"
                compile_project
                start_server || exit 1
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
        "threads_per_block": 512,
        "blocks": 1024,
        "ilp": 8,
        "persistent": true,
        "chunk_size": 131072,
        "binary_nonce": true,
        "progress_ms": 1000
    }'

    # 测试3: 优化配置 - ASCII模式
    run_test "优化-ASCII" '{
        "total_nonce": 10000000000,
        "threads_per_block": 512,
        "blocks": 1024,
        "ilp": 8,
        "persistent": true,
        "chunk_size": 131072,
        "binary_nonce": false,
        "odometer": true,
        "progress_ms": 1000
    }'

    # 测试4: 极限配置
    run_test "极限配置" '{
        "total_nonce": 10000000000,
        "threads_per_block": 256,
        "blocks": 2048,
        "ilp": 16,
        "persistent": true,
        "chunk_size": 262144,
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
            ;;
        quick)
            # 快速测试
            if ! check_server; then
                compile_project
                start_server || exit 1
            fi
            run_test "快速测试" '{
                "total_nonce": 1000000000,
                "threads_per_block": 512,
                "blocks": 1024,
                "ilp": 8,
                "persistent": true,
                "chunk_size": 262144,
                "binary_nonce": true
            }'
            ;;
        custom)
            # 自定义测试
            if [ -z "$2" ]; then
                echo "用法: $0 custom '{\"total_nonce\": 1000000000, ...}'"
                exit 1
            fi
            if ! check_server; then
                compile_project
                start_server || exit 1
            fi
            run_test "自定义测试" "$2"
            ;;
        *)
            echo "用法: $0 [compile|start|stop|test|quick|custom]"
            echo "  compile - 编译项目"
            echo "  start   - 编译并启动服务"
            echo "  stop    - 停止服务"
            echo "  test    - 运行完整测试（默认）"
            echo "  quick   - 运行快速测试"
            echo "  custom  - 运行自定义配置测试"
            exit 1
            ;;
    esac
}

# 运行主流程
main "$@"