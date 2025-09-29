#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 加速器选择（默认 GPU）
ACCEL_SELECTION="${1:-gpu}"
ACCEL_SELECTION=$(echo "$ACCEL_SELECTION" | tr '[:upper:]' '[:lower:]')

# 配置文件
PID_FILE="./bro-api.pid"
DEFAULT_LOG_FILE="./bro-api.log"
DEFAULT_PORT=8801

# 从环境变量或使用默认值
LOG_FILE="${LOG_FILE:-$DEFAULT_LOG_FILE}"
PORT="${PORT:-$DEFAULT_PORT}"

# 检查是否跳过编译（使用 SKIP_BUILD=1 ./start.sh 来跳过编译）
SKIP_BUILD="${SKIP_BUILD:-0}"

# 检查是否已在运行
if [ -f "$PID_FILE" ]; then
    if kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo -e "${YELLOW}BRO API is already running with PID $(cat $PID_FILE)${NC}"
        echo "Log file: $LOG_FILE"
        exit 1
    else
        # PID文件存在但进程不存在，清理PID文件
        rm -f "$PID_FILE"
    fi
fi

# 检查AUTH_TOKEN
if [ -z "$AUTH_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: AUTH_TOKEN not set, API will run without authentication!${NC}"
    echo -e "${YELLOW}Recommend: export AUTH_TOKEN=your-secret-token${NC}"
    read -p "Continue without authentication? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 设置二进制文件路径
BIN_PATH="./target/release/bro"

# 根据加速器选择配置内核构建与 Cargo 特性
declare -a CARGO_ARGS
declare -a KERNEL_TASKS

case "$ACCEL_SELECTION" in
    gpu|cuda)
        ACCEL_SELECTION="gpu"
        CARGO_ARGS=(build --release)
        KERNEL_TASKS+=("gpu")
        ;;
    fpga)
        CARGO_ARGS=(build --release --no-default-features --features fpga)
        KERNEL_TASKS+=("fpga")
        ;;
    both|all)
        ACCEL_SELECTION="both"
        CARGO_ARGS=(build --release --features fpga)
        KERNEL_TASKS+=("gpu" "fpga")
        ;;
    *)
        echo -e "${RED}Unknown accelerator option: $ACCEL_SELECTION${NC}"
        echo "请使用 ./start.sh [gpu|fpga|both]"
        exit 1
        ;;
esac

# 根据标志决定是否编译
if [ "$SKIP_BUILD" = "1" ]; then
    echo -e "${BLUE}Skipping compilation (SKIP_BUILD=1)${NC}"
    if [ ! -f "$BIN_PATH" ]; then
        echo -e "${RED}Binary not found at $BIN_PATH${NC}"
        echo -e "${YELLOW}Please run without SKIP_BUILD first to compile${NC}"
        exit 1
    fi
else
    # 自动编译以确保使用最新代码
    for task in "${KERNEL_TASKS[@]}"; do
        case "$task" in
            gpu)
                echo -e "${GREEN}Building CUDA kernel (build_cubin_ada.sh)...${NC}"
                if ! ./build_cubin_ada.sh; then
                    echo -e "${RED}CUDA kernel build failed${NC}"
                    exit 1
                fi
                ;;
            fpga)
                echo -e "${GREEN}Building FPGA bitstream (fpga/build.sh)...${NC}"
                if ! ./fpga/build.sh; then
                    echo -e "${RED}FPGA kernel build failed${NC}"
                    exit 1
                fi
                ;;
        esac
    done

    echo -e "${GREEN}Compiling latest code for ${ACCEL_SELECTION}...${NC}"
    echo -e "${YELLOW}(Use SKIP_BUILD=1 to skip compilation for faster restart)${NC}"
    if ! cargo "${CARGO_ARGS[@]}"; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Compilation successful${NC}"
fi

# 启动服务
echo -e "${GREEN}Starting BRO API Server (mode: ${ACCEL_SELECTION})...${NC}"
echo "Port: $PORT"
echo "Log file: $LOG_FILE"
if [ ! -z "$AUTH_TOKEN" ]; then
    echo "Auth: Enabled (token: ${AUTH_TOKEN:0:8}...)"
else
    echo "Auth: Disabled"
fi

# 创建日志文件（如果不存在）
touch "$LOG_FILE"

# 创建一个临时的启动脚本，确保所有输出都被捕获
cat > /tmp/bro-start-$$.sh << 'EOF'
#!/bin/bash
exec 1>> "$LOG_FILE"
exec 2>&1
exec "$BIN_PATH"
EOF
chmod +x /tmp/bro-start-$$.sh

# 后台启动，使用临时脚本确保所有输出都重定向
LOG_FILE="$LOG_FILE" \
AUTH_TOKEN="$AUTH_TOKEN" \
PORT="$PORT" \
BIN_PATH="$BIN_PATH" \
BRO_ACCELERATOR_MODE="$ACCEL_SELECTION" \
nohup /tmp/bro-start-$$.sh > /dev/null 2>&1 &

# 保存PID
PID=$!
echo $PID > "$PID_FILE"

# 等待1秒检查是否成功启动
sleep 1
if kill -0 $PID 2>/dev/null; then
    # 清理临时脚本
    rm -f /tmp/bro-start-$$.sh

    echo -e "${GREEN}BRO API started successfully with PID $PID${NC}"
    echo ""
    echo "Commands:"
    echo "  View logs:    tail -f $LOG_FILE"
    echo "  Stop server:  ./stop.sh"
    echo "  Check status: ./status.sh"
else
    echo -e "${RED}Failed to start BRO API${NC}"
    rm -f "$PID_FILE"
    rm -f /tmp/bro-start-$$.sh
    exit 1
fi
