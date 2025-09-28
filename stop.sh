#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID文件路径
PID_FILE="./bro-api.pid"

# 先尝试杀死所有GPU相关进程
echo -e "${BLUE}Checking for GPU processes...${NC}"
GPU_PIDS=$(pgrep -f "sha256_kernel|persistent_kernel" 2>/dev/null)
if [ ! -z "$GPU_PIDS" ]; then
    echo -e "${YELLOW}Found GPU kernel processes: $GPU_PIDS${NC}"
    for GPID in $GPU_PIDS; do
        echo -e "${YELLOW}Killing GPU process $GPID${NC}"
        kill -9 $GPID 2>/dev/null
    done
fi

# 检查nvidia-smi中的进程
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}Checking nvidia-smi for stuck processes...${NC}"
    # 获取所有使用GPU的bro相关进程
    NVIDIA_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ ! -z "$NVIDIA_PIDS" ]; then
        for NPID in $NVIDIA_PIDS; do
            # 检查是否是bro进程
            if ps -p $NPID -o comm= 2>/dev/null | grep -q "bro"; then
                echo -e "${YELLOW}Killing GPU process from nvidia-smi: $NPID${NC}"
                kill -9 $NPID 2>/dev/null
            fi
        done
    fi
fi

# 检查PID文件是否存在
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}PID file not found. Searching for running process...${NC}"

    # 尝试查找进程
    PID=$(pgrep -f "target/release/bro")
    if [ -z "$PID" ]; then
        echo -e "${YELLOW}BRO API main process is not running${NC}"
        # 仍然继续，因为可能有残留的GPU进程
    else
        echo -e "${GREEN}Found BRO API process with PID $PID${NC}"
    fi
else
    PID=$(cat "$PID_FILE")
fi

# 如果有PID，检查进程是否存在并停止
if [ ! -z "$PID" ]; then
    if ! kill -0 $PID 2>/dev/null; then
        echo -e "${YELLOW}Process $PID is not running${NC}"
        rm -f "$PID_FILE"
    else
        # 获取进程信息
        echo -e "${YELLOW}Stopping BRO API (PID: $PID)...${NC}"
        ps aux | grep $PID | grep -v grep

        # 发送SIGTERM信号（优雅关闭）
        kill $PID

        # 等待进程结束（最多10秒）
        COUNTER=0
        while kill -0 $PID 2>/dev/null && [ $COUNTER -lt 10 ]; do
            sleep 1
            COUNTER=$((COUNTER + 1))
            echo -n "."
        done
        echo ""

        # 如果进程仍在运行，强制终止
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}Process still running, forcing termination...${NC}"
            kill -9 $PID
            sleep 1
        fi

        # 最终检查
        if ! kill -0 $PID 2>/dev/null; then
            echo -e "${GREEN}BRO API main process stopped${NC}"
        else
            echo -e "${RED}Failed to stop BRO API${NC}"
            exit 1
        fi
    fi
fi

# 清理PID文件
rm -f "$PID_FILE"

# 最后再次检查是否有残留的bro进程
echo -e "${BLUE}Final cleanup check...${NC}"
REMAINING=$(pgrep -f "bro" | grep -v $$)
if [ ! -z "$REMAINING" ]; then
    echo -e "${YELLOW}Found remaining bro processes: $REMAINING${NC}"
    for RPID in $REMAINING; do
        echo -e "${YELLOW}Killing remaining process $RPID${NC}"
        kill -9 $RPID 2>/dev/null
    done
fi

echo -e "${GREEN}All BRO processes stopped successfully${NC}"