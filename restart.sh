#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Restarting BRO API Server...${NC}"

# 停止当前服务
if [ -f "./bro-api.pid" ]; then
    ./stop.sh
    echo ""
fi

# 启动新服务（会自动编译）
./start.sh

echo -e "${GREEN}Restart complete!${NC}"