# GPU优化方案指南

## 当前状况
- **现有性能**: 7.0-7.5 GH/s（SHA256双哈希）
- **硬件**: RTX 4090
- **算法限制**: SHA256在GPU上效率仅15-20%

## 已完成的优化

### 1. 创建了优化的CUDA内核
文件：`sha256_kernel_optimized_v2.cu`

**主要优化**：
- 批量处理：每个线程处理多个nonce（ILP=4）
- 共享内存缓存：减少全局内存访问
- 快速ASCII转换：使用查表法
- 循环展开：完全展开SHA256主循环
- 原子操作优化：减少全局原子操作

### 2. 创建了测试脚本
文件：`test_optimized.sh`
- 自动编译优化版本
- 测试多种配置
- 比较性能提升

## 如何使用

### 在Linux服务器上运行：

1. **推送代码到服务器**：
```bash
scp sha256_kernel_optimized_v2.cu user@server:/path/to/project/
scp test_optimized.sh user@server:/path/to/project/
```

2. **在服务器上编译**：
```bash
# 编译优化版本
nvcc -O3 \
  -arch=sm_89 \
  -maxrregcount=64 \
  -use_fast_math \
  -Xptxas -O3,-v \
  -Xptxas -dlcm=ca \
  -cubin sha256_kernel_optimized_v2.cu -o sha256_kernel.cubin

# 重启服务
pkill -f "target/release/bro"
cargo run --release &
```

3. **测试性能**：
```bash
./test_optimized.sh
```

## 预期性能提升

### SHA256优化（当前算法）
- **原始**: 7.0 GH/s
- **优化后**: 8.5-10 GH/s（提升20-40%）
- **极限**: 10-12 GH/s（需要手写PTX汇编）

### 为什么性能有限？
SHA256双哈希的固有问题：
1. **串行依赖**：第二次哈希依赖第一次结果
2. **内存带宽**：频繁的内存访问
3. **算法特性**：不适合GPU并行架构

## 突破性能瓶颈的方案

### 方案1：多GPU并行（推荐）
最简单有效，线性扩展性能：
- 2x RTX 4090 = 14-16 GH/s
- 4x RTX 4090 = 28-32 GH/s
- 8x RTX 4090 = 56-64 GH/s

实现方式：
```rust
// 修改配置支持多GPU
{
    "gpu_ids": [0, 1, 2, 3],  // 4张GPU
    "total_nonce": 1000000000000,
    "threads_per_block": 128,
    "blocks": 1024
}
```

### 方案2：更换算法（如果协议允许）

| 算法 | GPU效率 | RTX 4090性能 | 相比SHA256 |
|-----|---------|------------|-----------|
| **Ethash** | 90%+ | 140 MH/s | 20倍效率 |
| **KawPow** | 85%+ | 70 MH/s | 10倍效率 |
| **Autolykos2** | 80%+ | 600 MH/s | 85倍效率 |
| **Octopus** | 85%+ | 150 MH/s | 21倍效率 |

### 方案3：使用ASIC（终极方案）
如果必须使用SHA256：
- **Antminer S19 Pro**: 110 TH/s（是GPU的15,000倍）
- **Whatsminer M30S++**: 112 TH/s
- **成本效益比**: ASIC远超GPU

## 立即可执行的步骤

### 1. 恢复最佳配置（如果优化失败）
```bash
./restore_performance.sh
```

### 2. 测试优化版本
```bash
# 在GPU服务器上
./test_optimized.sh
```

### 3. 监控性能
```bash
# 实时监控GPU使用率
nvidia-smi dmon -i 0 -s pucvmet

# 查看功率和温度
watch -n 1 nvidia-smi
```

## 最终建议

1. **短期**（立即）:
   - 使用优化的内核，预期提升20-40%
   - 确保GPU运行在最高性能模式

2. **中期**（1周内）:
   - 实现多GPU并行
   - 考虑租用更多GPU

3. **长期**（评估）:
   - 如果持续需要高算力，考虑ASIC
   - 评估是否可以更换算法

## 性能基准

当前RTX 4090的SHA256性能已接近理论极限：
- **理论峰值**: 41.3 GH/s（82.6 TFLOPS / 2000 ops）
- **实际极限**: 10-12 GH/s（25-30%效率）
- **当前性能**: 7.0 GH/s（17%效率）
- **优化目标**: 8.5-10 GH/s（20-24%效率）

## 结论

当前7 GH/s已是SHA256在RTX 4090上的**合理性能**。优化版本可能提升到8.5-10 GH/s，但不会有数量级的提升。

要达到15 GH/s以上，唯一可行方案是：
1. 使用2张或更多GPU
2. 更换为GPU友好的算法
3. 使用ASIC硬件