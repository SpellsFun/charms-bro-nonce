#!/bin/bash

# 参数扫描脚本 - 找到最优配置

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== RTX 4090 参数扫描 ===${NC}\n"

# 结果文件
RESULTS="param_scan_results.csv"
echo "Config,Threads,Blocks,ILP,Chunk,GH/s" > $RESULTS

# 测试函数
test_params() {
    local threads=$1
    local blocks=$2
    local ilp=$3
    local chunk=$4

    local config="${threads}x${blocks}xILP${ilp}xC${chunk}"
    echo -ne "${YELLOW}测试 $config...${NC} "

    local outpoint="scan$(date +%s%N):1"

    local response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"$outpoint\",
            \"wait\": true,
            \"options\": {
                \"total_nonce\": 10000000000,
                \"threads_per_block\": $threads,
                \"blocks\": $blocks,
                \"ilp\": $ilp,
                \"persistent\": true,
                \"chunk_size\": $chunk,
                \"binary_nonce\": false,
                \"odometer\": true,
                \"progress_ms\": 0
            }
        }" 2>/dev/null)

    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo -e "${GREEN}${rate} GH/s${NC}"
        echo "$config,$threads,$blocks,$ilp,$chunk,$rate" >> $RESULTS
    else
        echo -e "${RED}失败${NC}"
    fi

    sleep 0.5
}

# 参数范围
THREADS=(128 256 512)
BLOCKS=(512 1024 2048 4096)
ILP=(4 8 16 32)
CHUNKS=(65536 131072 262144 524288)

echo "开始扫描..."
echo "线程数: ${THREADS[@]}"
echo "块数: ${BLOCKS[@]}"
echo "ILP: ${ILP[@]}"
echo "Chunk: ${CHUNKS[@]}"
echo ""

total=$((${#THREADS[@]} * ${#BLOCKS[@]} * ${#ILP[@]} * ${#CHUNKS[@]}))
count=0

# 扫描所有组合
for t in "${THREADS[@]}"; do
    for b in "${BLOCKS[@]}"; do
        for i in "${ILP[@]}"; do
            for c in "${CHUNKS[@]}"; do
                count=$((count + 1))
                echo -ne "[${count}/${total}] "
                test_params $t $b $i $c
            done
        done
    done
done

echo -e "\n${GREEN}=== 扫描完成 ===${NC}\n"

# 显示前10个最佳配置
echo -e "${BLUE}Top 10 配置:${NC}"
echo "排名 | 配置 | GH/s"
echo "------|------|------"
tail -n +2 $RESULTS | sort -t',' -k6 -rn | head -10 | nl | while IFS=',' read -r n config t b i c rate; do
    printf "%4s | %-20s | %.2f\n" "$n" "$(echo $config | cut -d',' -f1)" "$rate"
done

# 找出最佳配置
echo -e "\n${GREEN}最佳配置:${NC}"
best=$(tail -n +2 $RESULTS | sort -t',' -k6 -rn | head -1)
IFS=',' read -r config threads blocks ilp chunk rate <<< "$best"
echo "  配置: $config"
echo "  性能: ${rate} GH/s"
echo ""
echo "  threads_per_block: $threads"
echo "  blocks: $blocks"
echo "  ilp: $ilp"
echo "  chunk_size: $chunk"
echo ""
echo "完整结果保存在: $RESULTS"