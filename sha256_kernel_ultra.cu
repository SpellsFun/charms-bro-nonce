#include <stdint.h>
#include <string.h>

// SHA256常量表 - 放在常量内存中加速访问
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

// 两位数ASCII表 - 用于快速数字转换
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

// 使用PTX内联汇编的快速旋转
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

// CUDA设备字节交换 - 使用内建函数
__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

// SHA256压缩函数 - 极度优化版本
__device__ __noinline__ void sha256_transform(uint32_t state[8], const uint32_t data[16]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    // 使用寄存器数组存储W，减少内存访问
    uint32_t W[16];

    // 初始化W数组
    #pragma unroll 16
    for(int i = 0; i < 16; i++) {
        W[i] = data[i];
    }

    // SHA256主循环 - 完全展开
    #pragma unroll 64
    for (int t = 0; t < 64; t++) {
        uint32_t Wt;

        if (t < 16) {
            Wt = W[t];
        } else {
            // 消息扩展
            uint32_t w0 = W[t & 15];
            uint32_t w1 = W[(t-2) & 15];
            uint32_t w9 = W[(t-7) & 15];
            uint32_t w14 = W[(t-14) & 15];

            uint32_t s0 = rotr(w1, 7) ^ rotr(w1, 18) ^ (w1 >> 3);
            uint32_t s1 = rotr(w14, 17) ^ rotr(w14, 19) ^ (w14 >> 10);

            Wt = W[t & 15] = w0 + s0 + w9 + s1;
        }

        // SHA256轮函数
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + Kc[t] + Wt;

        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// 优化的uint64转ASCII - 使用查表法
__device__ int fast_u64_to_ascii(uint64_t value, char* buffer) {
    char temp[21];
    int len = 0;

    // 特殊处理0
    if (value == 0) {
        buffer[0] = '0';
        return 1;
    }

    // 两位一组处理，使用DIG2查表
    while (value >= 100) {
        int rem = value % 100;
        value /= 100;
        temp[len++] = DIG2[rem * 2 + 1];
        temp[len++] = DIG2[rem * 2];
    }

    // 处理剩余的1-2位
    if (value >= 10) {
        temp[len++] = DIG2[value * 2 + 1];
        temp[len++] = DIG2[value * 2];
    } else if (value > 0) {
        temp[len++] = '0' + value;
    }

    // 反转到正确顺序
    for (int i = 0; i < len; i++) {
        buffer[i] = temp[len - 1 - i];
    }

    return len;
}

// 计算前导零 - 使用内建函数
__device__ __forceinline__ uint32_t count_leading_zeros(const uint32_t hash[8]) {
    uint32_t lz = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t word = bswap32(hash[i]);
        if (word == 0) {
            lz += 32;
        } else {
            lz += __clz(word);
            break;
        }
    }

    return lz;
}

// 超级优化的持久化内核
extern "C" __global__ void __launch_bounds__(128, 8)
double_sha256_persistent_kernel(
    const uint8_t* __restrict__ outpoint_data,
    uint32_t outpoint_len,
    uint64_t* __restrict__ global_counter,
    uint64_t total_nonce,
    uint32_t* __restrict__ done,
    uint32_t* __restrict__ best_lz,
    uint64_t* __restrict__ best_nonce,
    uint32_t chunk_size,
    uint32_t binary_nonce,  // 始终忽略，使用ASCII
    uint32_t* __restrict__ odometer
) {
    // 动态共享内存 - 缓存outpoint数据
    extern __shared__ uint8_t shared_mem[];
    uint8_t* shared_outpoint = shared_mem;

    // 协作加载outpoint到共享内存
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // 每个线程负责加载一部分
    for (int i = tid; i < outpoint_len && i < 128; i += blockDim.x) {
        shared_outpoint[i] = outpoint_data[i];
    }
    __syncthreads();

    // 每个线程的局部最佳值
    uint32_t local_best_lz = 0;
    uint64_t local_best_nonce = 0;

    // ILP(指令级并行)因子
    const int ILP = 2;

    // 持久化内核主循环
    while (!(*done)) {
        // 原子获取下一批工作
        uint64_t work_start = atomicAdd((unsigned long long*)global_counter,
                                        (unsigned long long)chunk_size);

        if (work_start >= total_nonce) break;

        uint64_t work_end = min(work_start + chunk_size, total_nonce);

        // 处理分配的nonce范围
        for (uint64_t base_nonce = work_start;
             base_nonce < work_end;
             base_nonce += gridDim.x * blockDim.x * ILP) {

            // ILP展开 - 每个线程处理ILP个nonce
            #pragma unroll
            for (int ilp = 0; ilp < ILP; ilp++) {
                uint64_t nonce = base_nonce +
                                (bid * blockDim.x + tid) * ILP + ilp;

                if (nonce >= work_end) break;

                // ========== 第一次SHA256 ==========
                uint32_t state1[8] = {
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                };

                // 准备消息块
                uint8_t msg[64];
                memset(msg, 0, 64);

                // 复制outpoint (challenge = txid:vout)
                int pos = 0;
                for (; pos < outpoint_len && pos < 64; pos++) {
                    msg[pos] = shared_outpoint[pos];
                }

                // 添加ASCII nonce
                char nonce_str[21];
                int nonce_len = fast_u64_to_ascii(nonce, nonce_str);

                for (int i = 0; i < nonce_len && pos < 64; i++, pos++) {
                    msg[pos] = nonce_str[i];
                }

                int msg_len = pos;

                // 添加padding
                msg[msg_len] = 0x80;

                // 如果消息适合单个块（<= 55字节）
                if (msg_len <= 55) {
                    // 添加长度（大端）
                    uint64_t bit_len = (uint64_t)msg_len * 8;
                    for (int i = 0; i < 8; i++) {
                        msg[56 + i] = (bit_len >> (56 - i * 8)) & 0xff;
                    }

                    // 转换为32位字并字节交换
                    uint32_t W[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        W[i] = ((uint32_t)msg[i*4] << 24) |
                               ((uint32_t)msg[i*4+1] << 16) |
                               ((uint32_t)msg[i*4+2] << 8) |
                               ((uint32_t)msg[i*4+3]);
                        W[i] = bswap32(W[i]);
                    }

                    // 压缩
                    sha256_transform(state1, W);
                } else {
                    // 需要两个块
                    // 第一块
                    uint32_t W[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        W[i] = ((uint32_t)msg[i*4] << 24) |
                               ((uint32_t)msg[i*4+1] << 16) |
                               ((uint32_t)msg[i*4+2] << 8) |
                               ((uint32_t)msg[i*4+3]);
                        W[i] = bswap32(W[i]);
                    }
                    sha256_transform(state1, W);

                    // 第二块
                    memset(W, 0, sizeof(W));
                    uint64_t bit_len = (uint64_t)msg_len * 8;
                    W[14] = bswap32((uint32_t)(bit_len >> 32));
                    W[15] = bswap32((uint32_t)bit_len);
                    sha256_transform(state1, W);
                }

                // ========== 第二次SHA256 ==========
                uint32_t state2[8] = {
                    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
                };

                // 第二次哈希的输入是32字节
                uint32_t W2[16];
                memset(W2, 0, sizeof(W2));

                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    W2[i] = bswap32(state1[i]);
                }
                W2[8] = 0x80000000;  // padding
                W2[15] = bswap32(256);  // 长度: 256 bits

                sha256_transform(state2, W2);

                // 计算前导零
                uint32_t lz = count_leading_zeros(state2);

                // 更新局部最佳值
                if (lz > local_best_lz) {
                    local_best_lz = lz;
                    local_best_nonce = nonce;
                }
            }
        }

        // 定期更新odometer（进度指示器）
        if (odometer && tid == 0 && bid == 0) {
            atomicAdd(odometer, chunk_size);
        }
    }

    // 将局部最佳值更新到全局
    if (local_best_lz > 0) {
        atomicMax(best_lz, local_best_lz);
        if (local_best_lz == *best_lz) {
            atomicExch((unsigned long long*)best_nonce,
                      (unsigned long long)local_best_nonce);
        }
    }
}