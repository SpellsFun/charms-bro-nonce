#!/bin/bash

# RTX 4090 深度优化脚本

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== RTX 4090 深度优化 ===${NC}\n"

# 1. 系统优化
echo -e "${YELLOW}步骤1: 系统级优化${NC}"
echo "设置GPU最大性能..."
sudo nvidia-smi -pm 1 2>/dev/null || echo "需要sudo权限"
sudo nvidia-smi -pl 450 2>/dev/null || echo "功率限制设置失败"
sudo nvidia-smi -lgc 2520 2>/dev/null || echo "频率锁定失败"

# 2. 编译优化
echo -e "\n${YELLOW}步骤2: 重新编译优化内核${NC}"

# 创建优化的编译脚本
cat > build_optimized.sh << 'EOF'
#!/bin/bash
# RTX 4090优化编译参数
nvcc -O3 \
  -arch=sm_89 \
  -maxrregcount=96 \
  -use_fast_math \
  -Xptxas -O3,-v \
  -Xptxas -dlcm=ca \
  -Xptxas --optimize-float-atomics \
  -Xcompiler -O3,-march=native \
  -cubin sha256_kernel.cu -o sha256_kernel.cubin
EOF

chmod +x build_optimized.sh
./build_optimized.sh

# 3. 测试优化配置
echo -e "\n${YELLOW}步骤3: 测试优化配置${NC}"

test_config() {
    local name="$1"
    local threads="$2"
    local blocks="$3"
    local ilp="$4"
    local chunk="$5"

    local outpoint="opt4090_$(date +%s%N):1"

    echo -ne "${BLUE}$name: ${NC}"

    local response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"$outpoint\",
            \"wait\": true,
            \"options\": {
                \"total_nonce\": 50000000000,
                \"threads_per_block\": $threads,
                \"blocks\": $blocks,
                \"ilp\": $ilp,
                \"persistent\": true,
                \"chunk_size\": $chunk,
                \"binary_nonce\": false,
                \"odometer\": true,
                \"batch_size\": 100000000000,
                \"progress_ms\": 0
            }
        }")

    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo -e "${GREEN}${rate} GH/s${NC}"
        echo "$name|$threads|$blocks|$ilp|$chunk|$rate" >> opt_results.txt
    else
        echo -e "${RED}失败${NC}"
    fi

    sleep 1
}

# 清空结果
> opt_results.txt

# 重启服务
echo -e "\n${YELLOW}重启服务...${NC}"
pkill -f "target/release/bro" || true
sleep 2
cargo build --release
nohup cargo run --release > server.log 2>&1 &
sleep 3

echo -e "\n${YELLOW}开始优化测试...${NC}"

# 测试配置 - 基于RTX 4090的128个SM
# 16384 CUDA cores / 128 SM = 128 cores/SM
# 最优配置应该充分利用所有SM

echo -e "\n${BLUE}--- 高占用率配置 ---${NC}"
test_config "128x128_ILP16" 128 128 16 262144    # 1 block per SM
test_config "256x128_ILP8" 256 128 8 262144      # 1 block per SM, more threads
test_config "128x256_ILP16" 128 256 16 262144    # 2 blocks per SM
test_config "256x256_ILP8" 256 256 8 131072      # 2 blocks per SM

echo -e "\n${BLUE}--- 高并行配置 ---${NC}"
test_config "64x512_ILP32" 64 512 32 524288      # 4 blocks per SM
test_config "128x512_ILP16" 128 512 16 262144    # 4 blocks per SM
test_config "256x512_ILP8" 256 512 8 131072      # 4 blocks per SM

echo -e "\n${BLUE}--- 平衡配置 ---${NC}"
test_config "192x192_ILP12" 192 192 12 196608
test_config "384x128_ILP8" 384 128 8 131072
test_config "512x128_ILP4" 512 128 4 65536

echo -e "\n${BLUE}--- 极限配置 ---${NC}"
test_config "64x1024_ILP32" 64 1024 32 1048576
test_config "128x1024_ILP16" 128 1024 16 524288
test_config "256x1024_ILP8" 256 1024 8 262144

echo -e "\n${GREEN}=== 结果分析 ===${NC}"

# 显示Top 5
echo -e "\n${YELLOW}Top 5 配置:${NC}"
echo "配置|线程|块|ILP|Chunk|GH/s" | column -t -s'|'
echo "------|------|---|---|-------|-----"
sort -t'|' -k6 -rn opt_results.txt | head -5 | column -t -s'|'

# 最佳配置
echo -e "\n${GREEN}最佳配置:${NC}"
best=$(sort -t'|' -k6 -rn opt_results.txt | head -1)
IFS='|' read -r name threads blocks ilp chunk rate <<< "$best"
echo "  配置: $name"
echo "  性能: ${rate} GH/s"
echo "  参数:"
echo "    threads_per_block: $threads"
echo "    blocks: $blocks"
echo "    ilp: $ilp"
echo "    chunk_size: $chunk"

# 性能分析
echo -e "\n${YELLOW}性能分析:${NC}"
if (( $(echo "$rate > 10" | bc -l) )); then
    echo -e "${GREEN}✓ 达到10+ GH/s！${NC}"
elif (( $(echo "$rate > 8" | bc -l) )); then
    echo -e "${YELLOW}接近目标，继续优化${NC}"
else
    echo -e "${RED}性能仍需提升${NC}"
fi

echo -e "\n${BLUE}进一步优化建议:${NC}"
echo "1. 监控GPU利用率: nvidia-smi dmon -i 0"
echo "2. 检查温度: 保持<70°C获得最高频率"
echo "3. 尝试更激进的ILP: 64或128"
echo "4. 调整内存访问模式"