#!/bin/bash

# 日志文件路径
LOG_FILE="${LOG_FILE:-./bro-api.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

# 如果有参数，使用参数作为tail的行数
if [ ! -z "$1" ]; then
    tail -n "$1" "$LOG_FILE"
else
    # 默认实时查看
    tail -f "$LOG_FILE"
fi