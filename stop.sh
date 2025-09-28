#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# PID文件路径
PID_FILE="./bro-api.pid"

# 检查PID文件是否存在
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}PID file not found. Searching for running process...${NC}"

    # 尝试查找进程
    PID=$(pgrep -f "target/release/bro")
    if [ -z "$PID" ]; then
        echo -e "${RED}BRO API is not running${NC}"
        exit 1
    else
        echo -e "${GREEN}Found BRO API process with PID $PID${NC}"
    fi
else
    PID=$(cat "$PID_FILE")
fi

# 检查进程是否存在
if ! kill -0 $PID 2>/dev/null; then
    echo -e "${YELLOW}Process $PID is not running${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

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

# 清理PID文件
rm -f "$PID_FILE"

# 最终检查
if ! kill -0 $PID 2>/dev/null; then
    echo -e "${GREEN}BRO API stopped successfully${NC}"
else
    echo -e "${RED}Failed to stop BRO API${NC}"
    exit 1
fi