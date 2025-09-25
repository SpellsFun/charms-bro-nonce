# RTX 4090 性能调优指南

## 当前优化状态

### 已完成优化
1. **内核优化**
   - 使用 `__forceinline__` 优化关键函数
   - 调整 `__launch_bounds__` 提高占用率
   - 展开循环到16次迭代 (`#pragma unroll 16`)
   - 优化内存访问模式

2. **参数调优**
   - 线程块: 256 (从512降低，提高占用率)
   - 块数量: 4096 (从2048增加)
   - ILP: 16 (从8增加，最大化指令级并行)
   - Chunk大小: 262144 (翻倍)
   - 批处理: 50B (5倍增加)

3. **编译优化**
   - 寄存器数: 128
   - 快速数学: `-use_fast_math -ftz=true`
   - 优化级别: `-O3`

## 运行测试

### 1. 编译和启动
```bash
chmod +x run_optimized.sh
./run_optimized.sh
```

### 2. 性能测试（ASCII模式）
```bash
curl -X POST http://localhost:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "outpoint": "your_outpoint_here",
    "wait": true,
    "options": {
      "total_nonce": 100000000000,
      "threads_per_block": 256,
      "blocks": 4096,
      "ilp": 16,
      "persistent": true,
      "chunk_size": 262144,
      "batch_size": 50000000000,
      "progress_ms": 1000,
      "binary_nonce": false,
      "odometer": true
    }
  }'
```

### 3. 性能测试（二进制模式 - 更快）
```bash
curl -X POST http://localhost:8001/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "outpoint": "your_outpoint_here",
    "wait": true,
    "options": {
      "total_nonce": 100000000000,
      "threads_per_block": 256,
      "blocks": 4096,
      "ilp": 16,
      "persistent": true,
      "chunk_size": 262144,
      "batch_size": 50000000000,
      "progress_ms": 1000,
      "binary_nonce": true,
      "odometer": false
    }
  }'
```

## 进一步优化建议

### A. 尝试不同参数组合

1. **超大块配置** (GPU内存充足时)
```json
{
  "blocks": 8192,
  "chunk_size": 524288
}
```

2. **高ILP配置** (计算密集型)
```json
{
  "ilp": 32,
  "threads_per_block": 128
}
```

### B. 系统级优化

```bash
# 1. 设置GPU为最高性能模式
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 450  # RTX 4090最大功率

# 2. 锁定GPU频率（可选）
sudo nvidia-smi -lgc 2520  # 锁定到最高频率

# 3. 监控性能
watch -n 1 nvidia-smi
```

### C. 多GPU扩展

如果有多张RTX 4090：
```json
{
  "gpu_ids": [0, 1, 2, 3],
  "gpu_weights": [1.0, 1.0, 1.0, 1.0]
}
```

## 预期性能

| 配置 | 预期GH/s | 说明 |
|------|----------|------|
| 基础优化 | 8-10 | 当前已达到 |
| 中级优化 | 10-12 | 参数调优后 |
| 高级优化 | 12-15 | 系统级优化 |
| 极限优化 | 15-18 | 二进制模式+最优参数 |

## 性能验证

运行后查看输出：
```
[GPU 0] done: best_lz=XX nonce=XXXX
Summary: GPUs=1 elapsed XX.XXs, rate XX.XX GH/s
```

目标是看到 `rate` 达到 12-15 GH/s。

## 故障排除

1. **性能低于预期**
   - 检查GPU温度: `nvidia-smi -q -d TEMPERATURE`
   - 确认功率限制: `nvidia-smi -q -d POWER`
   - 验证频率: `nvidia-smi -q -d CLOCK`

2. **内存不足**
   - 减少blocks数量
   - 减小chunk_size

3. **编译失败**
   - 确保CUDA 12.0+
   - 检查架构支持: sm_89 for RTX 4090

## 注意事项

- **二进制模式更快**：约提升10-15%性能
- **ASCII模式兼容性更好**：与标准验证工具兼容
- **温度管理重要**：保持GPU温度<80°C
- **功率是关键**：确保供电充足（450W）