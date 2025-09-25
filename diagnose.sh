#!/bin/bash

# 诊断脚本 - 找出性能瓶颈

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== GPU性能诊断 ===${NC}\n"

# 1. GPU基本信息
echo -e "${YELLOW}1. GPU信息:${NC}"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory --format=csv

echo -e "\n${YELLOW}2. 性能状态:${NC}"
nvidia-smi -q -d PERFORMANCE | grep -E "Performance State|Graphics|SM Clock|Memory Clock|Applications Clock"

echo -e "\n${YELLOW}3. 功率管理:${NC}"
nvidia-smi -q -d POWER | grep -E "Power Draw|Power Limit|Default Power|Enforced Power"

echo -e "\n${YELLOW}4. 温度状态:${NC}"
nvidia-smi -q -d TEMPERATURE | grep -E "GPU Current|GPU Shutdown|GPU Slowdown"

echo -e "\n${YELLOW}5. 运行基准测试并监控:${NC}"
echo "启动GPU监控（10秒）..."

# 后台监控
nvidia-smi dmon -i 0 -s pucvmet -c 10 > gpu_monitor.txt &
MONITOR_PID=$!

# 运行测试
OUTPOINT="diagnose_$(date +%s):1"
echo "运行测试..."
RESULT=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d "{
    \"outpoint\": \"${OUTPOINT}\",
    \"wait\": true,
    \"options\": {
      \"total_nonce\": 50000000000,
      \"threads_per_block\": 256,
      \"blocks\": 2048,
      \"ilp\": 16,
      \"persistent\": true,
      \"chunk_size\": 262144,
      \"binary_nonce\": false,
      \"progress_ms\": 1000
    }
  }")

wait $MONITOR_PID

echo -e "\n${YELLOW}6. 测试结果:${NC}"
echo "$RESULT" | python3 -m json.tool | grep -E "rate_ghs|elapsed_secs"

echo -e "\n${YELLOW}7. GPU监控数据:${NC}"
echo "时间 功率(W) 温度(C) SM使用率(%) 内存使用率(%) SM频率(MHz) 内存频率(MHz)"
cat gpu_monitor.txt | tail -n +3

echo -e "\n${BLUE}=== 分析 ===${NC}"

# 分析功率
POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | cut -d'.' -f1)
if [ "$POWER" -lt 400 ]; then
    echo -e "${YELLOW}⚠ 功率较低 (${POWER}W)，可能被限制${NC}"
    echo "  建议: sudo nvidia-smi -pl 450"
fi

# 分析温度
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
if [ "$TEMP" -gt 80 ]; then
    echo -e "${YELLOW}⚠ 温度较高 (${TEMP}°C)，可能降频${NC}"
fi

# 分析利用率
UTIL=$(cat gpu_monitor.txt | tail -n +3 | awk '{sum+=$4; count++} END {print int(sum/count)}')
if [ "$UTIL" -lt 95 ]; then
    echo -e "${YELLOW}⚠ GPU利用率低 (平均${UTIL}%)，代码可能有瓶颈${NC}"
fi

echo -e "\n${GREEN}优化建议:${NC}"
echo "1. 设置最大性能模式:"
echo "   sudo nvidia-smi -pm 1"
echo "   sudo nvidia-smi -pl 450"
echo ""
echo "2. 锁定最高频率:"
echo "   sudo nvidia-smi -lgc 2520"
echo ""
echo "3. 如果还是6-7 GH/s，可能是："
echo "   - 代码已达到算法极限"
echo "   - 需要使用专门的SHA256优化库"
echo "   - 考虑使用多GPU并行"