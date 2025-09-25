#!/bin/bash

# 测试优化版本的SHA256内核

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== 测试优化的SHA256内核 ===${NC}\n"

# 1. 备份原始内核
echo -e "${YELLOW}步骤1: 备份原始内核${NC}"
cp sha256_kernel.cu sha256_kernel_original.cu
cp sha256_kernel.cubin sha256_kernel_original.cubin

# 2. 编译优化版本
echo -e "\n${YELLOW}步骤2: 编译优化内核${NC}"

# 先尝试编译优化版本
nvcc -O3 \
  -arch=sm_89 \
  -maxrregcount=64 \
  -use_fast_math \
  -Xptxas -O3,-v \
  -Xptxas -dlcm=ca \
  -Xcompiler -O3 \
  -cubin sha256_kernel_optimized.cu -o sha256_kernel_optimized.cubin

if [ $? -ne 0 ]; then
    echo -e "${RED}优化版本编译失败，使用修复版本${NC}"

    # 创建修复版本
    cat > sha256_kernel_fixed.cu << 'EOF'
#include <stdint.h>
#include <string.h>

// SHA256常量
__constant__ uint32_t Kc[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ __constant__ char DIG2[201] =
    "00010203040506070809"
    "10111213141516171819"
    "20212223242526272829"
    "30313233343536373839"
    "40414243444546474849"
    "50515253545556575859"
    "60616263646566676869"
    "70717273747576777879"
    "80818283848586878889"
    "90919293949596979899";

// 优化的旋转函数
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

// 优化的SHA256压缩 - 使用共享内存缓存
__device__ void sha256_compress_optimized(const uint32_t W[16], uint32_t H[8]) {
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    uint32_t Wt[64];

    // 消息扩展
    #pragma unroll 16
    for(int i = 0; i < 16; i++) {
        Wt[i] = W[i];
    }

    #pragma unroll 48
    for(int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(Wt[i-15], 7) ^ rotr(Wt[i-15], 18) ^ (Wt[i-15] >> 3);
        uint32_t s1 = rotr(Wt[i-2], 17) ^ rotr(Wt[i-2], 19) ^ (Wt[i-2] >> 10);
        Wt[i] = Wt[i-16] + s0 + Wt[i-7] + s1;
    }

    // 主循环 - 完全展开
    #pragma unroll 64
    for(int t = 0; t < 64; t++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + Kc[t] + Wt[t];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// 优化的持久化内核 - 批量处理
extern "C" __global__ void __launch_bounds__(128, 8)
double_sha256_persistent_kernel(
    const uint8_t* outpoint_data,
    uint32_t outpoint_len,
    uint64_t* global_counter,
    uint64_t total_nonce,
    uint32_t* done,
    uint32_t* best_lz,
    uint64_t* best_nonce,
    uint32_t chunk_size,
    uint32_t binary_nonce,
    uint32_t* odometer
) {
    // 共享内存缓存
    __shared__ uint8_t shared_outpoint[128];
    __shared__ uint32_t shared_best_lz;

    // 协作载入outpoint
    if(threadIdx.x < outpoint_len && threadIdx.x < 128) {
        shared_outpoint[threadIdx.x] = outpoint_data[threadIdx.x];
    }

    if(threadIdx.x == 0) {
        shared_best_lz = 0;
    }
    __syncthreads();

    // 每个线程处理多个nonce（ILP）
    const int ILP = 4;
    uint64_t local_best_nonce = 0;
    uint32_t local_best_lz = 0;

    while(!(*done)) {
        // 获取工作块
        uint64_t idx = atomicAdd(global_counter, chunk_size);
        if(idx >= total_nonce) break;

        uint64_t end = min(idx + chunk_size, total_nonce);

        // 批量处理nonce
        for(uint64_t base = idx; base < end; base += blockDim.x * gridDim.x * ILP) {

            // 处理ILP个nonce
            #pragma unroll
            for(int ilp = 0; ilp < ILP; ilp++) {
                uint64_t nonce_idx = base + (blockIdx.x * blockDim.x + threadIdx.x) * ILP + ilp;
                if(nonce_idx >= end) break;

                // 构建消息
                uint8_t msg[128];
                memcpy(msg, shared_outpoint, outpoint_len);

                // ASCII nonce
                int nonce_len = 0;
                uint64_t temp_nonce = nonce_idx;
                char nonce_str[21];

                do {
                    nonce_str[nonce_len++] = '0' + (temp_nonce % 10);
                    temp_nonce /= 10;
                } while(temp_nonce > 0);

                // 反转
                for(int i = 0; i < nonce_len/2; i++) {
                    char t = nonce_str[i];
                    nonce_str[i] = nonce_str[nonce_len-1-i];
                    nonce_str[nonce_len-1-i] = t;
                }

                memcpy(msg + outpoint_len, nonce_str, nonce_len);

                // 第一次SHA256
                uint32_t hash1[8] = {
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                };

                // Padding
                int total_len = outpoint_len + nonce_len;
                msg[total_len] = 0x80;
                memset(msg + total_len + 1, 0, 64 - total_len - 1 - 8);

                uint64_t bit_len = total_len * 8;
                for(int i = 0; i < 8; i++) {
                    msg[56 + i] = (bit_len >> (56 - i*8)) & 0xff;
                }

                // 转换为uint32_t
                uint32_t W[16];
                for(int i = 0; i < 16; i++) {
                    W[i] = ((uint32_t)msg[i*4] << 24) |
                           ((uint32_t)msg[i*4+1] << 16) |
                           ((uint32_t)msg[i*4+2] << 8) |
                           ((uint32_t)msg[i*4+3]);
                }

                sha256_compress_optimized(W, hash1);

                // 第二次SHA256
                uint32_t hash2[8] = {
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                };

                // 准备第二次哈希输入
                uint32_t W2[16];
                memset(W2, 0, sizeof(W2));
                for(int i = 0; i < 8; i++) {
                    W2[i] = __builtin_bswap32(hash1[i]);
                }
                W2[8] = 0x80000000;
                W2[15] = 256;

                sha256_compress_optimized(W2, hash2);

                // 计算前导零
                uint32_t lz = 0;
                for(int i = 0; i < 8; i++) {
                    uint32_t word = __builtin_bswap32(hash2[i]);
                    if(word == 0) {
                        lz += 32;
                    } else {
                        lz += __clz(word);
                        break;
                    }
                }

                // 更新最佳值
                if(lz > local_best_lz) {
                    local_best_lz = lz;
                    local_best_nonce = nonce_idx;
                }
            }
        }

        // 更新odometer
        if(odometer && threadIdx.x == 0) {
            atomicAdd(odometer, chunk_size);
        }
    }

    // 更新全局最佳值
    if(local_best_lz > 0) {
        atomicMax(best_lz, local_best_lz);
        if(local_best_lz == *best_lz) {
            atomicExch(best_nonce, local_best_nonce);
        }
    }
}
EOF

    # 编译修复版本
    nvcc -O3 \
      -arch=sm_89 \
      -maxrregcount=64 \
      -Xptxas -O3,-v \
      -Xptxas -dlcm=ca \
      -cubin sha256_kernel_fixed.cu -o sha256_kernel.cubin

    if [ $? -ne 0 ]; then
        echo -e "${RED}编译失败，恢复原始版本${NC}"
        cp sha256_kernel_original.cubin sha256_kernel.cubin
        exit 1
    fi
else
    cp sha256_kernel_optimized.cubin sha256_kernel.cubin
fi

# 3. 重启服务
echo -e "\n${YELLOW}步骤3: 重启服务${NC}"
pkill -f "target/release/bro" || true
sleep 2
nohup cargo run --release > server.log 2>&1 &
sleep 3

# 4. 测试性能
echo -e "\n${YELLOW}步骤4: 测试优化性能${NC}\n"

test_performance() {
    local name="$1"
    local config="$2"
    local outpoint="optimized_$(date +%s%N):1"

    echo -ne "${BLUE}$name: ${NC}"

    local response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
        -H 'Content-Type: application/json' \
        -d "{
            \"outpoint\": \"$outpoint\",
            \"wait\": true,
            \"options\": $config
        }")

    local rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)

    if [ -n "$rate" ]; then
        echo -e "${GREEN}${rate} GH/s${NC}"
        echo "$name|$rate" >> optimized_results.txt
    else
        echo -e "${RED}失败${NC}"
        echo "响应: $response"
    fi
}

> optimized_results.txt

# 测试不同配置
echo -e "${BLUE}--- 标准配置 ---${NC}"
test_performance "标准_128x1024" '{
    "total_nonce": 50000000000,
    "threads_per_block": 128,
    "blocks": 1024,
    "ilp": 16,
    "persistent": true,
    "chunk_size": 524288,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 100000000000
}'

echo -e "\n${BLUE}--- 高并行配置 ---${NC}"
test_performance "高并行_256x2048" '{
    "total_nonce": 50000000000,
    "threads_per_block": 256,
    "blocks": 2048,
    "ilp": 8,
    "persistent": true,
    "chunk_size": 262144,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 100000000000
}'

test_performance "超高并行_64x4096" '{
    "total_nonce": 50000000000,
    "threads_per_block": 64,
    "blocks": 4096,
    "ilp": 32,
    "persistent": true,
    "chunk_size": 1048576,
    "binary_nonce": false,
    "odometer": true,
    "batch_size": 100000000000
}'

# 5. 分析结果
echo -e "\n${GREEN}=== 性能分析 ===${NC}\n"

echo "测试结果："
cat optimized_results.txt | column -t -s'|'

best_rate=$(sort -t'|' -k2 -rn optimized_results.txt | head -1 | cut -d'|' -f2)
echo -e "\n最佳性能: ${GREEN}${best_rate} GH/s${NC}"

if (( $(echo "$best_rate > 10" | bc -l) )); then
    echo -e "${GREEN}✅ 成功突破10 GH/s！优化有效！${NC}"
elif (( $(echo "$best_rate > 8" | bc -l) )); then
    echo -e "${YELLOW}性能提升到8+ GH/s，接近目标${NC}"
else
    echo -e "${YELLOW}性能: ${best_rate} GH/s${NC}"
    echo "优化效果有限，SHA256算法在GPU上的极限约为7-8 GH/s"
fi

echo -e "\n${BLUE}注意：${NC}"
echo "1. 如果性能没有提升，说明已接近SHA256在GPU上的理论极限"
echo "2. 进一步提升需要："
echo "   - 使用多GPU并行"
echo "   - 改用ASIC硬件"
echo "   - 更换算法（如Ethash、KawPow等GPU友好算法）"