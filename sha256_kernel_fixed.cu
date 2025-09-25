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
