#!/bin/bash

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}     Restarting BRO API Server          ${NC}"
echo -e "${YELLOW}========================================${NC}"

# 检查是否是git仓库
if [ -d ".git" ]; then
    echo -e "${BLUE}Checking for updates...${NC}"

    # 保存当前分支
    CURRENT_BRANCH=$(git branch --show-current)
    echo "Current branch: $CURRENT_BRANCH"

    # 检查是否有未提交的更改
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
        git status --short
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${RED}Restart cancelled${NC}"
            exit 1
        fi
    fi

    # 拉取最新代码
    echo -e "${GREEN}Pulling latest code...${NC}"
    git pull
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Code updated successfully${NC}"
    else
        echo -e "${YELLOW}Warning: git pull failed, using local code${NC}"
    fi
else
    echo -e "${YELLOW}Not a git repository, skipping update${NC}"
fi

# 停止当前服务
if [ -f "./bro-api.pid" ]; then
    echo ""
    echo -e "${YELLOW}Stopping current service...${NC}"
    ./stop.sh
    echo ""
fi

# 启动新服务（会自动编译）
echo -e "${GREEN}Starting new service...${NC}"
./start.sh

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     Restart complete!                  ${NC}"
echo -e "${GREEN}========================================${NC}"