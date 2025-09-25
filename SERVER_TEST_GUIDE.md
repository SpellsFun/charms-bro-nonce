# 📋 服务器GPU优化测试指南

## 快速开始（5分钟）

### 1️⃣ 上传文件到服务器
```bash
# 从本地执行
scp sha256_kernel_ultra.cu root@your-server:~/charms-suite/charms-bro-nonce/
scp sha256_kernel_optimized_final.cu root@your-server:~/charms-suite/charms-bro-nonce/
scp server_test.sh root@your-server:~/charms-suite/charms-bro-nonce/
```

### 2️⃣ 登录服务器并测试
```bash
# SSH到服务器
ssh root@your-server

# 进入项目目录
cd ~/charms-suite/charms-bro-nonce

# 赋予执行权限
chmod +x server_test.sh

# 运行完整测试套件
./server_test.sh
```

## 手动测试步骤（如果自动脚本失败）

### 步骤1: 编译优化内核
```bash
# 编译超级优化版本
nvcc -O3 \
    -arch=sm_89 \
    -maxrregcount=64 \
    -use_fast_math \
    -Xptxas -O3,-v \
    -Xptxas -dlcm=ca \
    -Xcompiler -O3 \
    -cubin sha256_kernel_ultra.cu -o sha256_kernel.cubin
```

### 步骤2: 重启服务
```bash
# 停止现有服务
pkill -f "target/release/bro"

# 启动服务
nohup cargo run --release > server.log 2>&1 &

# 等待服务启动
sleep 3

# 检查服务状态
curl http://localhost:8001/api/v1/health
```

### 步骤3: 运行性能测试
```bash
# 标准测试
curl -X POST http://localhost:8001/api/v1/jobs \
    -H 'Content-Type: application/json' \
    -d '{
        "outpoint": "test_'$(date +%s)':1",
        "wait": true,
        "options": {
            "total_nonce": 100000000000,
            "threads_per_block": 128,
            "blocks": 1024,
            "ilp": 16,
            "persistent": true,
            "chunk_size": 524288,
            "binary_nonce": false,
            "odometer": true,
            "batch_size": 100000000000
        }
    }'
```

### 步骤4: 监控GPU状态
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或详细监控
nvidia-smi dmon -i 0 -s pucvmet
```

## 优化参数调整

### 🎯 最佳配置（RTX 4090）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| threads_per_block | 128 | 每个块的线程数 |
| blocks | 1024 | 总块数 |
| ilp | 16 | 指令级并行度 |
| chunk_size | 524288 | 工作块大小 |
| batch_size | 100000000000 | 批处理大小 |

### 🔧 参数微调指南

1. **如果GPU利用率低于90%**：
   - 增加 `blocks` 到 2048
   - 减少 `threads_per_block` 到 64

2. **如果出现内存错误**：
   - 减少 `chunk_size` 到 262144
   - 减少 `batch_size`

3. **如果温度过高（>80°C）**：
   - 限制功率：`sudo nvidia-smi -pl 400`

## 性能基准

### RTX 4090 预期性能

| 版本 | 预期性能 | 说明 |
|------|----------|------|
| 原始版本 | 6.5-7.5 GH/s | 未优化 |
| 优化版本 | 8.0-9.0 GH/s | 标准优化 |
| 超级优化 | 9.0-10.5 GH/s | 极限优化 |

### 性能瓶颈分析

```bash
# 检查瓶颈
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv

# 解读：
# GPU利用率 < 90%: 代码优化不足
# 内存利用率 > 80%: 内存带宽瓶颈
# 温度 > 80°C: 可能降频
# 功率 < 400W: 功率限制
```

## 故障排除

### ❌ 编译错误
```bash
# 检查CUDA版本
nvcc --version

# 如果版本低于11.0，使用兼容参数
nvcc -O3 -arch=sm_86 -cubin sha256_kernel_ultra.cu -o sha256_kernel.cubin
```

### ❌ 服务无法启动
```bash
# 检查端口占用
lsof -i:8001

# 查看错误日志
tail -f server.log

# 手动启动调试
cargo run --release
```

### ❌ 性能低于预期
```bash
# 1. 设置最高性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 450

# 2. 关闭其他GPU进程
nvidia-smi
# 找到占用GPU的进程并kill

# 3. 清理GPU缓存
sudo nvidia-smi -r

# 4. 确保使用正确的内核
ls -la *.cubin
file sha256_kernel.cubin
```

## 多GPU配置（如有多张卡）

```bash
# 查看所有GPU
nvidia-smi -L

# 设置使用特定GPU
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1

# 未来可修改代码支持多GPU并行
```

## 性能对比命令

```bash
# 快速对比测试
for config in "128 1024 16" "256 2048 8" "64 2048 32"; do
    set -- $config
    echo "测试配置: threads=$1 blocks=$2 ilp=$3"

    curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d '{
            "outpoint": "bench_'$(date +%s%N)':1",
            "wait": true,
            "options": {
                "total_nonce": 50000000000,
                "threads_per_block": '$1',
                "blocks": '$2',
                "ilp": '$3',
                "persistent": true,
                "chunk_size": 524288,
                "binary_nonce": false,
                "odometer": true,
                "batch_size": 100000000000
            }
        }' | grep -o '"rate_ghs":[0-9.]*'

    sleep 2
done
```

## 最终建议

### ✅ 立即可做
1. 运行 `server_test.sh` 获得完整测试报告
2. 使用最佳配置部署
3. 监控GPU状态确保稳定运行

### 📈 性能提升路径
1. **当前优化**: 7 GH/s → 9-10 GH/s (提升30-40%)
2. **双GPU并行**: 18-20 GH/s
3. **四GPU集群**: 36-40 GH/s
4. **ASIC方案**: 100+ TH/s

### ⚠️ 注意事项
- SHA256双哈希在GPU上的理论极限约为10-12 GH/s
- 进一步提升需要硬件扩展或ASIC
- 确保散热良好，温度控制在75°C以下

## 联系支持

如遇到问题：
1. 保存 `server.log` 和 `test_results.txt`
2. 记录 `nvidia-smi` 输出
3. 提供错误信息用于调试

---

**预计测试时间**: 10-15分钟
**预期性能提升**: 20-40%
**最佳性能目标**: 9-10 GH/s