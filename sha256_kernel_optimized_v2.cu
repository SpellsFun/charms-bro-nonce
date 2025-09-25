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

// ASCII数字表
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

// 计算前导零（word级别）
__device__ uint32_t count_leading_zeros_words(const uint32_t hash[8]) {
    uint32_t lz = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (hash[i] == 0) {
            lz++;
        } else {
            break;
        }
    }
    return lz;
}

// 优化的SHA256压缩函数
__device__ void sha256_compress_words(const uint32_t W0[16], uint32_t H[8]) {
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    uint32_t W[16];
    #pragma unroll
    for(int i = 0; i < 16; i++) {
        W[i] = W0[i];
    }

    #pragma unroll
    for (int t = 0; t < 64; t++) {
        uint32_t Wt;
        if (t < 16) {
            Wt = W[t];
        } else {
            uint32_t s0 = rotr(W[(t + 1) & 15], 7) ^ rotr(W[(t + 1) & 15], 18) ^ (W[(t + 1) & 15] >> 3);
            uint32_t s1 = rotr(W[(t + 14) & 15], 17) ^ rotr(W[(t + 14) & 15], 19) ^ (W[(t + 14) & 15] >> 10);
            Wt = W[t & 15] = W[t & 15] + s0 + W[(t + 9) & 15] + s1;
        }

        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + Kc[t] + Wt;
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

// 快速uint64转ASCII
__device__ int fast_uitoa(uint64_t n, char* s) {
    char tmp[21];
    int len = 0;

    // 使用查表法加速
    while (n >= 100) {
        uint32_t rem = n % 100;
        n /= 100;
        tmp[len++] = DIG2[rem * 2 + 1];
        tmp[len++] = DIG2[rem * 2];
    }

    if (n >= 10) {
        tmp[len++] = DIG2[n * 2 + 1];
        tmp[len++] = DIG2[n * 2];
    } else {
        tmp[len++] = '0' + n;
    }

    // 反转字符串
    for (int i = 0; i < len; i++) {
        s[i] = tmp[len - 1 - i];
    }

    return len;
}

// 优化的持久化内核 - 每个线程处理多个nonce
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
    uint32_t binary_nonce,  // 忽略，始终使用ASCII
    uint32_t* odometer
) {
    // 共享内存缓存
    __shared__ uint8_t shared_outpoint[80];
    __shared__ uint32_t shared_state[64];  // 缓存中间状态

    // 载入outpoint到共享内存
    if (threadIdx.x < outpoint_len && threadIdx.x < 80) {
        shared_outpoint[threadIdx.x] = outpoint_data[threadIdx.x];
    }
    __syncthreads();

    // 局部变量
    uint32_t local_best_lz = 0;
    uint64_t local_best_nonce = 0;

    // ILP展开因子
    const int ILP = 4;

    while (!(*done)) {
        // 原子获取下一个工作块 - 正确的类型转换
        uint64_t idx = atomicAdd((unsigned long long*)global_counter, (unsigned long long)chunk_size);
        if (idx >= total_nonce) break;

        uint64_t end = min(idx + chunk_size, total_nonce);

        // 批量处理nonce
        for (uint64_t base = idx; base < end; base += blockDim.x * gridDim.x * ILP) {

            // 使用ILP处理多个nonce
            #pragma unroll
            for (int ilp = 0; ilp < ILP; ilp++) {
                uint64_t nonce_val = base + (blockIdx.x * blockDim.x + threadIdx.x) * ILP + ilp;
                if (nonce_val >= end) break;

                // === 第一次SHA256 ===
                uint32_t H1[8] = {
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                };

                // 构建第一个消息块
                uint32_t W[16];
                memset(W, 0, sizeof(W));

                // 复制outpoint（字节转换为字）
                for (int i = 0; i < outpoint_len && i < 64; i++) {
                    ((uint8_t*)W)[i] = shared_outpoint[i];
                }

                // 快速ASCII nonce生成
                char nonce_str[21];
                int nonce_len = fast_uitoa(nonce_val, nonce_str);

                // 将nonce添加到消息
                for (int i = 0; i < nonce_len && (outpoint_len + i) < 64; i++) {
                    ((uint8_t*)W)[outpoint_len + i] = nonce_str[i];
                }

                // 添加padding
                int msg_len = outpoint_len + nonce_len;
                ((uint8_t*)W)[msg_len] = 0x80;

                // 如果消息短于56字节，在同一块中添加长度
                if (msg_len < 56) {
                    uint64_t bit_len = msg_len * 8;
                    W[14] = bit_len >> 32;
                    W[15] = bit_len & 0xffffffff;
                }

                // 转换为大端
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    W[i] = __builtin_bswap32(W[i]);
                }

                // 压缩
                sha256_compress_words(W, H1);

                // 如果消息超过56字节，需要第二个块
                if (msg_len >= 56) {
                    memset(W, 0, sizeof(W));
                    if (msg_len < 64) {
                        // 继续padding
                        for (int i = msg_len + 1; i < 64; i++) {
                            ((uint8_t*)W)[i - 64] = 0;
                        }
                    }
                    uint64_t bit_len = msg_len * 8;
                    W[14] = __builtin_bswap32(bit_len >> 32);
                    W[15] = __builtin_bswap32(bit_len & 0xffffffff);
                    sha256_compress_words(W, H1);
                }

                // === 第二次SHA256 ===
                uint32_t H2[8] = {
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                };

                // 第二次哈希的输入是第一次的输出（32字节）
                memset(W, 0, sizeof(W));
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    W[i] = __builtin_bswap32(H1[i]);
                }
                W[8] = 0x80000000;  // padding
                W[15] = __builtin_bswap32(256);  // 长度：256位

                sha256_compress_words(W, H2);

                // 计算前导零（以32位字为单位）
                uint32_t lz = count_leading_zeros_words(H2) * 32;

                // 如果不是完整的32位零，计算实际的位数
                if (lz < 256) {
                    uint32_t word = __builtin_bswap32(H2[lz / 32]);
                    if (word != 0) {
                        lz += __clz(word);
                    }
                }

                // 更新局部最佳值
                if (lz > local_best_lz) {
                    local_best_lz = lz;
                    local_best_nonce = nonce_val;
                }
            }
        }

        // 更新odometer
        if (odometer && threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(odometer, chunk_size);
        }
    }

    // 更新全局最佳值
    if (local_best_lz > 0) {
        atomicMax(best_lz, local_best_lz);
        // 如果我们有最佳值，更新nonce
        if (local_best_lz == *best_lz) {
            atomicExch((unsigned long long*)best_nonce, (unsigned long long)local_best_nonce);
        }
    }
}