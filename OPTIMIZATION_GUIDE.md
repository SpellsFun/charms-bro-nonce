# GPU性能优化指南 - RTX 4090

## 已完成的优化

### 1. CUDA内核参数优化
- **线程块大小**: 256 → 512 线程/块
- **网格大小**: 1024 → 2048 块
- **ILP(指令级并行)**: 1 → 8
- **Chunk大小**: 65536 → 131072
- **批处理大小**: 1B → 10B

### 2. 内存访问优化
- **缓存配置**: PreferL1 → PreferShared (更好的共享内存利用)
- **强制内联优化**: 添加`__forceinline__`到关键函数
- **循环展开**: 优化`#pragma unroll`指令

### 3. 编译优化
- **寄存器数量**: 64 → 128 (充分利用RTX 4090的寄存器)
- **快速数学**: 启用`-use_fast_math -ftz=true`
- **占用率**: __launch_bounds__(512, 2) → (512, 1)提高占用率

## 运行建议

### 基础运行命令
```bash
cargo build --release
cargo run --release -- <outpoint>
```

### 高性能运行参数
```bash
# 使用优化参数运行
cargo run --release -- <outpoint> \
  --threads-per-block 512 \
  --blocks 2048 \
  --ilp 8 \
  --persistent \
  --chunk-size 131072 \
  --batch-size 10000000000
```

### 环境变量优化
```bash
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB CUDA缓存
export CUDA_FORCE_PTX_JIT=1           # 强制PTX JIT编译
```

## 性能预期

基于RTX 4090的规格和优化：
- **理论峰值**: ~20 GH/s (SHA256双哈希)
- **实际预期**: 12-15 GH/s (考虑内存带宽限制)
- **相比原始7GH/s**: 提升约70-115%

## 进一步优化建议

1. **多GPU并行**: 如果有多张4090，使用`--gpu-ids`参数
2. **动态调整**: 监控GPU温度，动态调整频率
3. **内存池**: 考虑实现自定义内存池减少分配开销

## 编译注意事项

确保安装CUDA工具链后重新编译：
```bash
# 为RTX 4090 (SM 8.9)编译
ARCH=sm_89 ./build_cubin_ada.sh

# 或使用PTX fallback
nvcc -O3 -ptx -arch=compute_89 sha256_kernel.cu -o sha256_kernel.ptx
```

## 验证性能

运行后查看输出中的GH/s指标，应该能达到12-15 GH/s的目标。