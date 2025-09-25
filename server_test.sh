#!/bin/bash

# 服务器端GPU优化测试脚本
# 在RTX 4090服务器上运行

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      RTX 4090 SHA256 优化测试套件           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}\n"

# 1. 系统检查
echo -e "${CYAN}=== 步骤1: 系统环境检查 ===${NC}"
echo -n "检查CUDA版本: "
nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//'
echo -n "检查GPU: "
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

# 2. 备份当前内核
echo -e "${CYAN}=== 步骤2: 备份当前内核 ===${NC}"
if [ -f sha256_kernel.cubin ]; then
    cp sha256_kernel.cubin sha256_kernel.backup.$(date +%s).cubin
    echo "✓ 已备份当前内核"
else
    echo "⚠ 未找到现有内核文件"
fi
echo ""

# 3. 编译所有版本
echo -e "${CYAN}=== 步骤3: 编译优化版本 ===${NC}"

compile_kernel() {
    local src=$1
    local out=$2
    local desc=$3
    local flags=$4

    echo -ne "${YELLOW}编译 $desc...${NC}"

    if nvcc -O3 -arch=sm_89 $flags -cubin $src -o $out 2>/tmp/nvcc_error.log; then
        echo -e " ${GREEN}✓ 成功${NC}"
        return 0
    else
        echo -e " ${RED}✗ 失败${NC}"
        cat /tmp/nvcc_error.log | head -5
        return 1
    fi
}

# 编译原始版本（如果存在）
if [ -f sha256_kernel.cu ]; then
    compile_kernel "sha256_kernel.cu" "original.cubin" "原始版本" \
        "-maxrregcount=64 -Xptxas -O3,-v -Xptxas -dlcm=ca"
fi

# 编译超级优化版本
if [ -f sha256_kernel_ultra.cu ]; then
    compile_kernel "sha256_kernel_ultra.cu" "ultra.cubin" "超级优化版" \
        "-maxrregcount=64 -use_fast_math -Xptxas -O3,-v -Xptxas -dlcm=ca -Xcompiler -O3"
fi

# 编译最终优化版本
if [ -f sha256_kernel_optimized_final.cu ]; then
    compile_kernel "sha256_kernel_optimized_final.cu" "final.cubin" "最终优化版" \
        "-maxrregcount=64 -use_fast_math -Xptxas -O3,-v -Xptxas -dlcm=ca"
fi

echo ""

# 4. 性能测试函数
test_performance() {
    local cubin_file=$1
    local test_name=$2
    local config=$3

    # 使用指定的cubin
    cp $cubin_file sha256_kernel.cubin

    # 重启服务
    pkill -f "target/release/bro" 2>/dev/null || true
    sleep 2
    nohup cargo run --release > /tmp/bro.log 2>&1 &
    local server_pid=$!
    sleep 3

    # 确认服务启动
    if ! kill -0 $server_pid 2>/dev/null; then
        echo -e "${RED}服务启动失败${NC}"
        return 1
    fi

    # 创建唯一的outpoint
    local outpoint="test_$(date +%s%N):1"

    # 执行测试
    local response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"$outpoint\",
            \"wait\": true,
            \"options\": $config
        }" 2>/dev/null)

    # 提取性能数据
    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)
    local elapsed=$(echo "$response" | grep -o '"elapsed_secs":[0-9.]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo "$test_name|$cubin_file|$rate|$elapsed" >> test_results.txt
        printf "%-20s: ${GREEN}%8.2f GH/s${NC} (%6.2fs)\n" "$test_name" "$rate" "$elapsed"
    else
        echo "$test_name|$cubin_file|FAILED|0" >> test_results.txt
        printf "%-20s: ${RED}失败${NC}\n" "$test_name"
    fi

    # 停止服务
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
}

# 5. 执行测试
echo -e "${CYAN}=== 步骤4: 性能测试 ===${NC}\n"

# 清空结果文件
> test_results.txt

# 测试配置
DEFAULT_CONFIG='{
    "total_nonce": 100000000000,
    "threads_per_block": 128,
    "blocks": 1024,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 100000000000
}'

OPTIMIZED_CONFIG='{
    "total_nonce": 100000000000,
    "threads_per_block": 256,
    "blocks": 2048,
    "ilp": 8,
    "persistent": true,
    "chunk_size": 262144,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 200000000000
}'

echo -e "${BLUE}--- 标准配置测试 ---${NC}"
if [ -f original.cubin ]; then
    test_performance "original.cubin" "原始版本" "$DEFAULT_CONFIG"
fi
if [ -f final.cubin ]; then
    test_performance "final.cubin" "最终优化版" "$DEFAULT_CONFIG"
fi
if [ -f ultra.cubin ]; then
    test_performance "ultra.cubin" "超级优化版" "$DEFAULT_CONFIG"
fi

echo -e "\n${BLUE}--- 优化配置测试 ---${NC}"
if [ -f ultra.cubin ]; then
    test_performance "ultra.cubin" "超优+优配" "$OPTIMIZED_CONFIG"
fi

echo ""

# 6. 结果分析
echo -e "${CYAN}=== 步骤5: 结果分析 ===${NC}\n"

if [ -f test_results.txt ]; then
    echo -e "${YELLOW}测试结果汇总:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-20s %-15s %10s %10s\n" "测试名称" "内核版本" "性能(GH/s)" "时间(s)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    while IFS='|' read -r name kernel rate time; do
        if [ "$rate" = "FAILED" ]; then
            printf "%-20s %-15s ${RED}%10s${NC} %10s\n" "$name" "$(basename $kernel .cubin)" "$rate" "$time"
        else
            printf "%-20s %-15s ${GREEN}%10.2f${NC} %10.2f\n" "$name" "$(basename $kernel .cubin)" "$rate" "$time"
        fi
    done < test_results.txt

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 找出最佳配置
    echo -e "\n${GREEN}最佳性能:${NC}"
    best=$(sort -t'|' -k3 -rn test_results.txt | head -1)
    IFS='|' read -r name kernel rate time <<< "$best"
    echo "  配置: $name"
    echo "  内核: $(basename $kernel .cubin)"
    echo "  性能: ${rate} GH/s"

    # 性能提升计算
    if [ -f original.cubin ]; then
        original_rate=$(grep "original.cubin" test_results.txt | head -1 | cut -d'|' -f3)
        if [ -n "$original_rate" ] && [ "$original_rate" != "FAILED" ]; then
            improvement=$(echo "scale=1; ($rate - $original_rate) * 100 / $original_rate" | bc)
            echo -e "  提升: ${GREEN}+${improvement}%${NC}"
        fi
    fi
fi

echo ""

# 7. GPU状态监控
echo -e "${CYAN}=== 步骤6: GPU状态 ===${NC}"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.sm --format=csv
echo ""

# 8. 建议
echo -e "${CYAN}=== 优化建议 ===${NC}"

# 获取最佳性能值
best_rate=$(sort -t'|' -k3 -rn test_results.txt 2>/dev/null | head -1 | cut -d'|' -f3)

if [ -z "$best_rate" ] || [ "$best_rate" = "FAILED" ]; then
    echo -e "${RED}⚠ 所有测试失败，请检查环境配置${NC}"
elif (( $(echo "$best_rate > 10" | bc -l) )); then
    echo -e "${GREEN}✅ 优秀！已达到10+ GH/s${NC}"
    echo "   这是RTX 4090在SHA256双哈希上的极佳性能"
elif (( $(echo "$best_rate > 8" | bc -l) )); then
    echo -e "${YELLOW}✓ 良好！达到8+ GH/s${NC}"
    echo "   已接近RTX 4090的理论极限"
    echo "   建议："
    echo "   1. 确保GPU功率限制: sudo nvidia-smi -pl 450"
    echo "   2. 监控温度保持<70°C"
else
    echo -e "${YELLOW}⚠ 性能${best_rate} GH/s${NC}"
    echo "   可能的优化："
    echo "   1. 检查GPU利用率: nvidia-smi dmon -i 0"
    echo "   2. 调整线程和块配置"
    echo "   3. 确保使用优化的内核版本"
fi

echo ""
echo -e "${BLUE}进一步提升方案:${NC}"
echo "1. 多GPU并行: 2×RTX 4090 = 14-16 GH/s"
echo "2. ASIC矿机: 100+ TH/s (15000倍提升)"
echo ""

# 9. 清理
echo -e "${CYAN}=== 清理 ===${NC}"
echo -n "是否恢复原始内核? (y/n): "
read -r answer
if [ "$answer" = "y" ]; then
    if [ -f sha256_kernel.backup.*.cubin ]; then
        latest_backup=$(ls -t sha256_kernel.backup.*.cubin | head -1)
        cp $latest_backup sha256_kernel.cubin
        echo "✓ 已恢复原始内核"
    fi
fi

echo ""
echo -e "${GREEN}测试完成！${NC}"
echo "结果已保存到: test_results.txt"