#!/bin/bash

# 恢复RTX 4090最佳性能配置

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== 恢复RTX 4090最佳配置 ===${NC}\n"

# 1. 移除频率锁定，恢复自动boost
echo -e "${YELLOW}步骤1: 移除频率锁定${NC}"
sudo nvidia-smi -rgc 2>/dev/null && echo "✓ 移除频率锁定，恢复自动boost" || echo "需要sudo权限"
sudo nvidia-smi -pm 1 2>/dev/null && echo "✓ 持久模式开启"
sudo nvidia-smi -pl 450 2>/dev/null && echo "✓ 功率限制450W"

# 2. 恢复最佳编译参数
echo -e "\n${YELLOW}步骤2: 恢复最佳编译参数${NC}"
cat > build_cubin_ada.sh << 'EOF'
#!/bin/bash
# 恢复到工作良好的配置
nvcc -O3 \
  -arch=sm_89 \
  -maxrregcount=64 \
  -Xptxas -O3,-v \
  -Xptxas -dlcm=ca \
  -cubin sha256_kernel.cu -o sha256_kernel.cubin
EOF

chmod +x build_cubin_ada.sh
./build_cubin_ada.sh

# 3. 恢复CUDA kernel配置
echo -e "\n${YELLOW}步骤3: 恢复kernel配置${NC}"

# 恢复launch_bounds到合理值
sed -i.bak 's/__launch_bounds__([0-9]*, [0-9]*)/__launch_bounds__(128, 8)/' sha256_kernel.cu

# 4. 恢复Rust默认参数到最佳值
echo -e "\n${YELLOW}步骤4: 恢复默认参数${NC}"
cat > restore_lib.rs << 'EOF'
// 恢复到稳定的7.2 GH/s配置
pub const DEFAULT_THREADS_PER_BLOCK: u32 = 128;
pub const DEFAULT_BLOCKS: u32 = 1024;
pub const DEFAULT_ILP: u32 = 16;
pub const DEFAULT_CHUNK_SIZE: u32 = 524_288;
pub const DEFAULT_BATCH_SIZE: u64 = 100_000_000_000;
EOF

# 应用更改
sed -i.bak 's/pub const DEFAULT_THREADS_PER_BLOCK: u32 = .*/pub const DEFAULT_THREADS_PER_BLOCK: u32 = 128;/' src/lib.rs
sed -i.bak 's/pub const DEFAULT_BLOCKS: u32 = .*/pub const DEFAULT_BLOCKS: u32 = 1024;/' src/lib.rs
sed -i.bak 's/pub const DEFAULT_ILP: u32 = .*/pub const DEFAULT_ILP: u32 = 16;/' src/lib.rs
sed -i.bak 's/pub const DEFAULT_CHUNK_SIZE: u32 = .*/pub const DEFAULT_CHUNK_SIZE: u32 = 524_288;/' src/lib.rs

# 恢复缓存配置到PreferL1
sed -i.bak 's/CacheConfig::PreferShared/CacheConfig::PreferL1/' src/lib.rs

# 5. 重新编译
echo -e "\n${YELLOW}步骤5: 重新编译${NC}"
cargo build --release

# 6. 重启服务
echo -e "\n${YELLOW}步骤6: 重启服务${NC}"
pkill -f "target/release/bro" || true
sleep 2
nohup cargo run --release > server.log 2>&1 &
sleep 3

# 7. 测试恢复后的性能
echo -e "\n${YELLOW}步骤7: 测试性能${NC}"
OUTPOINT="restore_$(date +%s):1"

response=$(curl -s -X POST http://localhost:8001/api/v1/jobs \
    -H 'Content-Type: application/json' \
    -d "{
        \"outpoint\": \"$OUTPOINT\",
        \"wait\": true,
        \"options\": {
            \"total_nonce\": 50000000000,
            \"threads_per_block\": 128,
            \"blocks\": 1024,
            \"ilp\": 16,
            \"persistent\": true,
            \"chunk_size\": 524288,
            \"binary_nonce\": false,
            \"odometer\": true,
            \"batch_size\": 100000000000
        }
    }")

rate=$(echo "$response" | grep -o '"rate_ghs":[0-9.]*' | cut -d: -f2)
echo -e "\n恢复后性能: ${GREEN}${rate} GH/s${NC}"

# 8. GPU状态
echo -e "\n${YELLOW}GPU状态:${NC}"
nvidia-smi --query-gpu=clocks.current.sm,temperature.gpu,power.draw --format=csv

echo -e "\n${GREEN}=== 恢复完成 ===${NC}"
echo "预期性能: 7.0-7.5 GH/s"
echo "如果性能正常，说明之前的'优化'确实是错误的"