#!/bin/bash

# 部署和测试脚本
# 用于将代码部署到服务器并运行性能测试

set -e

# 配置
REMOTE_HOST="${REMOTE_HOST:-root@your-server.com}"
REMOTE_DIR="${REMOTE_DIR:-/root/charms-suite/charms-bro-nonce}"
LOCAL_DIR="."

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== 部署和测试脚本 ===${NC}"

# Step 1: 同步代码到服务器
sync_code() {
    echo -e "\n${YELLOW}1. 同步代码到服务器...${NC}"

    # 排除不必要的文件
    rsync -avz --exclude='target/' \
               --exclude='*.js' \
               --exclude='.git/' \
               --exclude='*.log' \
               --exclude='benchmark_results.csv' \
               ./ "${REMOTE_HOST}:${REMOTE_DIR}/"

    echo -e "${GREEN}代码同步完成${NC}"
}

# Step 2: 在服务器上编译
remote_compile() {
    echo -e "\n${YELLOW}2. 远程编译...${NC}"

    ssh "${REMOTE_HOST}" << 'EOF'
cd /root/charms-suite/charms-bro-nonce

# 设置环境变量
export CUDA_CACHE_MAXSIZE=4294967296
export CUDA_FORCE_PTX_JIT=1

# 编译CUDA内核
echo "编译CUDA内核..."
ARCH=sm_89 RREG=128 ./build_cubin_ada.sh

# 编译Rust
echo "编译Rust代码..."
cargo build --release

echo "编译完成"
EOF

    echo -e "${GREEN}远程编译完成${NC}"
}

# Step 3: 启动服务
start_service() {
    echo -e "\n${YELLOW}3. 启动服务...${NC}"

    ssh "${REMOTE_HOST}" << 'EOF'
cd /root/charms-suite/charms-bro-nonce

# 停止旧服务
pkill -f "target/release/bro" || true
sleep 2

# 启动新服务
nohup cargo run --release > server.log 2>&1 &
echo "服务启动中..."
sleep 3

# 检查服务状态
if pgrep -f "target/release/bro" > /dev/null; then
    echo "服务启动成功"
else
    echo "服务启动失败"
    tail -n 20 server.log
    exit 1
fi
EOF

    echo -e "${GREEN}服务启动成功${NC}"
}

# Step 4: 运行基准测试
run_benchmark() {
    echo -e "\n${YELLOW}4. 运行性能测试...${NC}"

    ssh "${REMOTE_HOST}" << 'EOF'
cd /root/charms-suite/charms-bro-nonce

# 运行基准测试
chmod +x benchmark.sh

echo "运行快速测试..."
./benchmark.sh quick

echo ""
echo "运行完整基准测试..."
./benchmark.sh

# 显示结果
echo ""
echo "=== 测试结果 ==="
cat benchmark_results.csv
EOF

    # 获取结果文件
    echo -e "\n${YELLOW}获取测试结果...${NC}"
    scp "${REMOTE_HOST}:${REMOTE_DIR}/benchmark_results.csv" ./remote_benchmark_results.csv

    echo -e "${GREEN}测试完成，结果已保存到: remote_benchmark_results.csv${NC}"
}

# Step 5: 显示优化建议
show_recommendations() {
    echo -e "\n${GREEN}=== 优化建议 ===${NC}"

    if [ -f "remote_benchmark_results.csv" ]; then
        # 分析结果
        best_rate=$(tail -n +2 remote_benchmark_results.csv | sort -t',' -k2 -rn | head -1 | cut -d',' -f2)

        echo "当前最佳性能: ${best_rate} GH/s"

        # 根据性能给出建议
        if (( $(echo "$best_rate < 10" | bc -l) )); then
            echo -e "${YELLOW}性能低于预期，建议:${NC}"
            echo "1. 增加blocks数量: --blocks 8192"
            echo "2. 调整ILP: --ilp 32"
            echo "3. 检查GPU温度和频率"
        elif (( $(echo "$best_rate < 15" | bc -l) )); then
            echo -e "${YELLOW}性能良好，可尝试:${NC}"
            echo "1. 使用二进制模式获得更高性能"
            echo "2. 增加chunk_size: --chunk-size 524288"
            echo "3. 调整线程数: --threads-per-block 128"
        else
            echo -e "${GREEN}性能优秀！已达到目标${NC}"
        fi
    fi
}

# 主流程
main() {
    case "${1:-all}" in
        sync)
            sync_code
            ;;
        compile)
            remote_compile
            ;;
        start)
            start_service
            ;;
        test)
            run_benchmark
            show_recommendations
            ;;
        all)
            sync_code
            remote_compile
            start_service
            run_benchmark
            show_recommendations
            ;;
        *)
            echo "用法: $0 [all|sync|compile|start|test]"
            echo "  all     - 执行所有步骤（默认）"
            echo "  sync    - 仅同步代码"
            echo "  compile - 仅编译"
            echo "  start   - 仅启动服务"
            echo "  test    - 仅运行测试"
            exit 1
            ;;
    esac
}

# 运行
main "$@"