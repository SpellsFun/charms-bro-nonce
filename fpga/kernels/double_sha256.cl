__constant uint K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint rotr32(uint x, uint n) {
    uint s = n & 31u;
    if (s == 0u) {
        return x;
    }
    return (x >> s) | (x << (32u - s));
}

inline void sha256_compress(const uchar block[64], uint state[8]) {
    uint w[64];
#pragma HLS ARRAY_PARTITION variable=w complete dim=1
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint)block[i * 4] << 24) | ((uint)block[i * 4 + 1] << 16) |
               ((uint)block[i * 4 + 2] << 8) | ((uint)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint a = state[0];
    uint b = state[1];
    uint c = state[2];
    uint d = state[3];
    uint e = state[4];
    uint f = state[5];
    uint g = state[6];
    uint h = state[7];

    for (int i = 0; i < 64; ++i) {
#pragma HLS PIPELINE II=1
        uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint temp1 = h + S1 + ch + K256[i] + w[i];
        uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

inline void sha256_digest(const uchar *msg, uint len, uchar out[32]) {
    uint state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint total_blocks = (len + 9 + 63) / 64;
    ulong bit_len = ((ulong)len) * 8ul;

    for (uint blk = 0; blk < total_blocks; ++blk) {
        uchar block[64];
#pragma HLS ARRAY_PARTITION variable=block complete dim=1
        for (int i = 0; i < 64; ++i) {
            ulong idx = ((ulong)blk) * 64ul + (ulong)i;
            uchar value = (uchar)0;
            if (idx < len) {
                value = msg[idx];
            } else if (idx == len) {
                value = (uchar)0x80;
            } else {
                ulong pad_start = (ulong)total_blocks * 64ul - 8ul;
                if (idx >= pad_start) {
                    ulong shift = (pad_start + 7ul) - idx;
                    value = (uchar)((bit_len >> (shift * 8ul)) & 0xfful);
                } else {
                    value = 0;
                }
            }
            block[i] = value;
        }
        sha256_compress(block, state);
    }

    for (int i = 0; i < 8; ++i) {
        out[i * 4] = (uchar)((state[i] >> 24) & 0xffu);
        out[i * 4 + 1] = (uchar)((state[i] >> 16) & 0xffu);
        out[i * 4 + 2] = (uchar)((state[i] >> 8) & 0xffu);
        out[i * 4 + 3] = (uchar)(state[i] & 0xffu);
    }
}

inline uint count_leading_zero_bits(const uchar digest[32]) {
    uint zeros = 0;
    for (int i = 0; i < 8; ++i) {
        uint word = ((uint)digest[i * 4] << 24) |
                    ((uint)digest[i * 4 + 1] << 16) |
                    ((uint)digest[i * 4 + 2] << 8) |
                    ((uint)digest[i * 4 + 3]);
        if (word == 0u) {
            zeros += 32u;
        } else {
            zeros += clz(word);
            break;
        }
    }
    return zeros;
}

inline uint encode_nonce_decimal(ulong nonce, uchar *out_digits) {
    if (nonce == 0ul) {
        out_digits[0] = (uchar)'0';
        return 1u;
    }
    uchar tmp[32];
    uint count = 0u;
    ulong value = nonce;
    while (value > 0ul) {
        tmp[count++] = (uchar)('0' + (value % 10ul));
        value /= 10ul;
    }
    for (uint i = 0; i < count; ++i) {
        out_digits[i] = tmp[count - 1u - i];
    }
    return count;
}

__kernel void double_sha256_fpga(
    __constant uchar *base,
    const uint base_len,
    const ulong start_nonce,
    const ulong total_nonce,
    const uint binary_nonce,
    __global uint *best_lz_out,
    __global ulong *best_nonce_out)
{
    uchar message[160];
#pragma HLS ARRAY_PARTITION variable=message complete dim=1
    uchar first_hash[32];
    uchar second_hash[32];
    uint local_best = 0u;
    ulong local_nonce = start_nonce;

    for (ulong offset = 0ul; offset < total_nonce; ++offset) {
#pragma HLS PIPELINE II=1
        ulong nonce = start_nonce + offset;
        for (uint i = 0u; i < base_len; ++i) {
            message[i] = base[i];
        }
        uint msg_len = base_len;

        if (binary_nonce != 0u) {
            ulong temp = nonce;
            for (uint b = 0u; b < 8u; ++b) {
                message[msg_len + b] = (uchar)((temp >> (b * 8u)) & 0xfful);
            }
            msg_len += 8u;
        } else {
            uchar digits[32];
            uint digit_count = encode_nonce_decimal(nonce, digits);
            for (uint d = 0u; d < digit_count; ++d) {
                message[msg_len + d] = digits[d];
            }
            msg_len += digit_count;
        }

        sha256_digest(message, msg_len, first_hash);
        sha256_digest(first_hash, 32u, second_hash);

        uint zeros = count_leading_zero_bits(second_hash);
        if (zeros > local_best) {
            local_best = zeros;
            local_nonce = nonce;
        }
    }

    best_lz_out[0] = local_best;
    best_nonce_out[0] = local_nonce;
}
