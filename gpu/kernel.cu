extern "C" {

#include <stdint.h>

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n){ return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z){ return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z){ return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t BSIG0(uint32_t x){ return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22); }
__device__ __forceinline__ uint32_t BSIG1(uint32_t x){ return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25); }
__device__ __forceinline__ uint32_t SSIG0(uint32_t x){ return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t SSIG1(uint32_t x){ return rotr(x,17) ^ rotr(x,19) ^ (x >> 10); }

__constant__ uint32_t K[64] = {
  0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
  0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
  0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
  0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
  0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92E,0x92722c85U,
  0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
  0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
  0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
};

struct Params {
  const uint8_t* challenge;   // challenge bytes
  uint32_t       challenge_len;
  uint64_t       start_nonce;
  uint32_t       count;       // number of nonces (threads)
  uint32_t       ilp;         // 每个线程处理的 nonce 数
  // output
  uint32_t*      best_digest; // 8 * u32
  uint32_t*      best_info;   // best_lz, best_nonce_lo, best_nonce_hi, lock
};

// 简单 64位 /10 除法，返回 商和 余数
__device__ __forceinline__ void u64_divmod10(uint64_t x, uint64_t* q, uint32_t* r){
  *q = x / 10ULL;
  *r = (uint32_t)(x - (*q)*10ULL);
}

// 把 u64 转十进制 ASCII，返回长度（无前导0）
__device__ __forceinline__ uint32_t u64_to_dec(uint64_t n, uint8_t* out /*至少20字节*/){
  if (n == 0ULL) { out[0] = '0'; return 1; }
  uint8_t tmp[20];
  uint32_t len = 0;
  while (n != 0ULL){
    uint64_t q; uint32_t r;
    u64_divmod10(n, &q, &r);
    tmp[len++] = (uint8_t)('0' + r);
    n = q;
  }
  // 倒过来写到 out
  for (uint32_t i = 0; i < len; ++i) out[i] = tmp[len-1-i];
  return len;
}

// 单块压缩
__device__ void sha256_compress(const uint32_t w0_15[16], uint32_t state[8]){
  uint32_t w[64];
  #pragma unroll
  for (int i=0;i<16;i++) w[i]=w0_15[i];
  #pragma unroll
  for (int t=16;t<64;t++){
    w[t] = SSIG1(w[t-2]) + w[t-7] + SSIG0(w[t-15]) + w[t-16];
  }
  uint32_t a=state[0], b=state[1], c=state[2], d=state[3];
  uint32_t e=state[4], f=state[5], g=state[6], h=state[7];
  #pragma unroll
  for (int t=0;t<64;t++){
    uint32_t T1 = h + BSIG1(e) + Ch(e,f,g) + K[t] + w[t];
    uint32_t T2 = BSIG0(a) + Maj(a,b,c);
    h=g; g=f; f=e; e=d + T1; d=c; c=b; b=a; a=T1 + T2;
  }
  state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
  state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__device__ void sha256_init(uint32_t h[8]){
  h[0]=0x6a09e667U; h[1]=0xbb67ae85U; h[2]=0x3c6ef372U; h[3]=0xa54ff53aU;
  h[4]=0x510e527fU; h[5]=0x9b05688cU; h[6]=0x1f83d9abU; h[7]=0x5be0cd19U;
}

__device__ void sha256_block(const uint32_t w0_15[16], uint32_t out[8]){
  sha256_init(out);
  sha256_compress(w0_15, out);
}

// 对 32 字节 digest 再做一次 sha256（固定长度，恰好单块）
__device__ void sha256_of_digest(const uint32_t digest[8], uint32_t out[8]){
  uint32_t w[16];
  #pragma unroll
  for (int i=0;i<8;i++) w[i]=digest[i];
  w[8]=0x80000000U;
  #pragma unroll
  for (int i=9;i<15;i++) w[i]=0;
  w[15]=256U; // 32 bytes = 256 bits
  sha256_block(w, out);
}

// 统计前导 0 bit（大端）
__device__ __forceinline__ uint32_t leading_zeros_256(const uint32_t h[8]){
  uint32_t lz = 0;
  #pragma unroll
  for (int i=0;i<8;i++){
    uint32_t x = h[i];
    if (x==0){ lz += 32; continue; }
    // 找到最高的1之前的连续0
    // __clz(x) 返回前导 0 个数（基于 32位）
    lz += __clz(x);
    break;
  }
  return lz;
}

// 把 msg = challenge || nonceASCII 做成 SHA256（可跨多块）
// 为简洁，这里直接按通用填充走（长度<= challenge_len + 20, 足以 1~2 块）
__device__ void sha256_double(const uint8_t* challenge, uint32_t clen, uint64_t nonce, uint32_t out[8]){
  // 先把 challenge 与 nonce ascii 拼到一个最多 (clen+20) 的临时缓冲
  // 为了性能，尽量栈上操作；4090 的寄存/栈足够
  uint8_t bufA[256]; // 假设 challenge 不超过 236 字节（足够 "${64hex}:${vout}" ）
  #pragma unroll
  for (uint32_t i=0;i<clen;i++) bufA[i]=challenge[i];
  uint8_t* p = bufA + clen;
  uint8_t nonce_ascii[20];
  uint32_t nlen = u64_to_dec(nonce, nonce_ascii);
  #pragma unroll
  for (uint32_t i=0;i<nlen;i++) p[i]=nonce_ascii[i];
  uint32_t total = clen + nlen;

  // 通用 SHA256：可能需要 1 或 2 块
  uint64_t bitlen = (uint64_t)total * 8ULL;
  // 写到最多两个 64字节的块里
  // 第一块 w0_15
  uint32_t w[16]; uint32_t h1[8];

  // 先清空
  #pragma unroll
  for (int i=0;i<16;i++) w[i]=0;

  // 填前 64 字节
  int i=0;
  for (; i<total && i<64; ++i){
    int wi = i >> 2;
    w[wi] = (w[wi] << 8) | bufA[i];
  }
  if (i < 64){
    // 写 0x80
    int wi = i >> 2;
    w[wi] = (w[wi] << 8) | 0x80;
    ++i;
    // 若还能继续在首块写长度（最后 8 字节），则在 w[14], w[15]
    if (i <= 56){
      // 填零直到 56
      for (; i<56; ++i){
        int wi2 = i >> 2; w[wi2] = (w[wi2] << 8);
      }
      // 写 64-bit 大端长度
      w[14] = (uint32_t)(bitlen >> 32);
      w[15] = (uint32_t)(bitlen & 0xffffffffU);
      sha256_block(w, h1);
    } else {
      // 首块放不下长度，需要两块
      // 填零到末尾
      for (; i<64; ++i){
        int wi2 = i >> 2; w[wi2] = (w[wi2] << 8);
      }
      uint32_t htmp[8];
      sha256_block(w, htmp);
      // 第二块全 0，再写长度
      uint32_t w2[16];
      #pragma unroll
      for (int k=0;k<16;k++) w2[k]=0;
      w2[14] = (uint32_t)(bitlen >> 32);
      w2[15] = (uint32_t)(bitlen & 0xffffffffU);

      // 继续压缩
      for (int j=0;j<8;j++) h1[j]=htmp[j];
      sha256_compress(w2, h1);
    }
  } else {
    // i==64 说明首块刚好装满原文；需写第二块 0x80 与长度
    uint32_t w2[16];
    #pragma unroll
    for (int k=0;k<16;k++) w2[k]=0;
    // 第一字节 0x80
    w2[0] = 0x80000000U;
    // 长度
    w2[14] = (uint32_t)(bitlen >> 32);
    w2[15] = (uint32_t)(bitlen & 0xffffffffU);
    sha256_block(w, h1);
    sha256_compress(w2, h1);
  }

  // 双哈希
  sha256_of_digest(h1, out);
}

__global__ void mine_kernel(Params p){
  uint64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ilp = p.ilp == 0 ? 1 : p.ilp;
  uint64_t base_nonce = p.start_nonce + thread_idx * (uint64_t)ilp;
  uint64_t start_index = thread_idx * (uint64_t)ilp;

  for (uint32_t i = 0; i < ilp; ++i){
    uint64_t global_index = start_index + i;
    if (global_index >= p.count) return;
    uint64_t nonce = base_nonce + i;

    uint32_t h2[8];
    sha256_double(p.challenge, p.challenge_len, nonce, h2);
    uint32_t lz = leading_zeros_256(h2);

    // 原子：如果 lz 更大则更新
    // best_info: [best_lz, nonce_lo, nonce_hi, lock]
    uint32_t* info = p.best_info;
    uint32_t prev = atomicMax(info + 0, lz);
    if (lz > prev){
      // 写 digest 与 nonce（无需复杂自旋锁，竞争概率低；如需严格并发一致，可用CAS+锁）
      for (int k=0;k<8;k++) p.best_digest[k] = h2[k];
      uint32_t lo = (uint32_t)(nonce & 0xffffffffULL);
      uint32_t hi = (uint32_t)((nonce >> 32) & 0xffffffffULL);
      info[1] = lo; info[2] = hi;
    }
  }
}

// 供 Rust 侧 launch 的元信息
__host__ void launch_mine(void* stream,
                          const uint8_t* challenge, uint32_t clen,
                          uint64_t start_nonce, uint32_t count,
                          uint32_t ilp,
                          uint32_t* best_digest, uint32_t* best_info,
                          int blocks, int threads){
  Params p { challenge, clen, start_nonce, count, ilp, best_digest, best_info };
  mine_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(p);
}

} // extern "C"
