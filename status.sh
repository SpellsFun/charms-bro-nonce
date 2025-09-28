#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 文件路径
PID_FILE="./bro-api.pid"
LOG_FILE="${LOG_FILE:-./bro-api.log}"

echo "========================================="
echo "         BRO API Server Status           "
echo "========================================="

# 检查PID文件
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo -e "PID File:     ${GREEN}Found${NC} (PID: $PID)"

    # 检查进程是否运行
    if kill -0 $PID 2>/dev/null; then
        echo -e "Process:      ${GREEN}Running${NC}"

        # 显示进程详情
        echo ""
        echo "Process Details:"
        ps -p $PID -o pid,vsz,rss,pcpu,pmem,comm,args

        # 显示端口监听情况
        echo ""
        echo "Listening Ports:"
        netstat -tlnp 2>/dev/null | grep $PID || lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep $PID || echo "  (需要root权限查看端口)"
    else
        echo -e "Process:      ${RED}Not Running${NC} (PID文件过期)"
    fi
else
    echo -e "PID File:     ${RED}Not Found${NC}"

    # 尝试查找进程
    PIDS=$(pgrep -f "target/release/bro")
    if [ ! -z "$PIDS" ]; then
        echo -e "Process:      ${YELLOW}Found without PID file${NC}"
        echo "Found PIDs:   $PIDS"
    else
        echo -e "Process:      ${RED}Not Running${NC}"
    fi
fi

# 检查日志文件
echo ""
if [ -f "$LOG_FILE" ]; then
    SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    LINES=$(wc -l < "$LOG_FILE")
    echo -e "Log File:     ${GREEN}Found${NC} ($LOG_FILE)"
    echo "  Size:       $SIZE"
    echo "  Lines:      $LINES"
    echo ""
    echo "Last 5 log entries:"
    echo "-----------------------------------------"
    tail -5 "$LOG_FILE"
else
    echo -e "Log File:     ${RED}Not Found${NC} ($LOG_FILE)"
fi

# 检查API健康状态
echo ""
echo "API Health Check:"
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    # 尝试访问API
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8801/api/v1/jobs 2>/dev/null)
    if [ "$RESPONSE" = "401" ]; then
        echo -e "  API Status: ${GREEN}Running${NC} (Authentication required)"
    elif [ "$RESPONSE" = "200" ]; then
        echo -e "  API Status: ${GREEN}Running${NC} (No authentication)"
    elif [ ! -z "$RESPONSE" ]; then
        echo -e "  API Status: ${YELLOW}Responding${NC} (HTTP $RESPONSE)"
    else
        echo -e "  API Status: ${RED}Not Responding${NC}"
    fi
else
    echo -e "  API Status: ${RED}Server Not Running${NC}"
fi

echo "========================================="