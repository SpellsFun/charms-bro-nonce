# 算法改进方案

## 当前问题
SHA256双哈希在GPU上效率低（仅15-20%），因为：
- 串行依赖性强
- 内存访问模式不友好
- 无法充分利用GPU并行性

## 推荐GPU友好算法

### 1. **Ethash/Etchash** (最推荐)
- **效率**: GPU利用率90%+
- **性能**: RTX 4090可达120-140 MH/s
- **特点**: 内存密集型，充分利用GPU带宽
- **代币**: ETH Classic, Ravencoin等

### 2. **KawPow**
- **效率**: GPU利用率85%+
- **性能**: RTX 4090可达60-70 MH/s
- **特点**: 专为GPU设计，ASIC抗性
- **代币**: Ravencoin (RVN)

### 3. **Autolykos2**
- **效率**: GPU利用率80%+
- **性能**: RTX 4090可达500-600 MH/s
- **特点**: 内存带宽友好
- **代币**: Ergo (ERG)

### 4. **Octopus**
- **效率**: GPU利用率85%+
- **性能**: RTX 4090可达130-150 MH/s
- **特点**: 多算法组合
- **代币**: Conflux (CFX)

## 算法改进实现示例

### 方案1: 批量并行处理
```cuda
// 同时处理多个独立的nonce范围
__global__ void batch_sha256_kernel(
    uint8_t* headers,      // 多个区块头
    uint32_t* nonce_starts, // 每个区块的起始nonce
    uint32_t* results,      // 结果数组
    uint32_t batch_size     // 批量大小
) {
    // 每个block处理一个独立的区块头
    // 避免串行依赖
}
```

### 方案2: 混合算法
```cuda
// 结合内存密集型操作
__global__ void hybrid_hash_kernel(
    uint8_t* data,
    uint32_t* dag,         // 大型DAG数组
    uint32_t* results
) {
    // 1. 内存密集型预处理（GPU友好）
    // 2. SHA256最终哈希（必要时）
    // 3. 内存访问模式优化
}
```

### 方案3: SIMD优化
```cuda
// 使用向量化指令
__global__ void vectorized_sha256_kernel() {
    uint4 data = make_uint4(...);  // 128位操作
    // 使用PTX内联汇编进行SIMD操作
    asm("vadd.u32.u32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
}
```

## 性能对比

| 算法 | RTX 4090性能 | GPU效率 | 内存使用 |
|-----|------------|---------|---------|
| SHA256双哈希 | 7 GH/s | 15-20% | 低 |
| Ethash | 140 MH/s | 90%+ | 高(4GB+) |
| KawPow | 70 MH/s | 85%+ | 中(2GB) |
| Autolykos2 | 600 MH/s | 80%+ | 中(2GB) |

## 实施建议

1. **短期方案**: 使用恢复脚本回到7.2 GH/s稳定性能
2. **中期方案**: 实现批量并行处理，预期10-12 GH/s
3. **长期方案**: 切换到GPU友好算法，效率提升4-5倍

## 代码修改示例

如需改用Ethash算法：
```rust
// src/lib.rs
pub enum Algorithm {
    SHA256Double,  // 当前
    Ethash,        // GPU友好
    KawPow,        // GPU优化
}

impl Algorithm {
    pub fn hash(&self, data: &[u8]) -> Vec<u8> {
        match self {
            Algorithm::SHA256Double => self.sha256_double(data),
            Algorithm::Ethash => self.ethash_hash(data),
            Algorithm::KawPow => self.kawpow_hash(data),
        }
    }
}
```

## 结论

1. SHA256双哈希在GPU上已接近极限（7 GH/s）
2. 继续"优化"只会适得其反
3. 改用GPU友好算法可获得5-10倍性能提升
4. 如必须用SHA256，考虑ASIC或多GPU并行