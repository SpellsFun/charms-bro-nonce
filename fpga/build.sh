#!/usr/bin/env bash
set -euo pipefail

# 简单的 Vitis/v++ 编译脚本，负责把 OpenCL 核心打包成 .xclbin。
# 需要提前 source AWS shell 与 XRT/Vitis 环境变量。

usage() {
    cat <<USAGE
用法: PLATFORM=<aws_shell> ./fpga/build.sh [输出文件]

环境变量：
  PLATFORM   必填，AWS Shell 的平台文件，例如：
             /opt/aws/platform/aws-vu9p-f1-04261818/xdma/feature_aws_vnc_2/
  VPP        可选，v++ 路径（默认直接调用 v++）
  TARGET     可选，编译目标：hw | hw_emu | sw_emu（默认 hw）
  KERNEL     可选，内核源码路径（默认 fpga/kernels/double_sha256.cl）
  FREQ_MHZ   可选，期望频率（MHz），通过 --kernel_frequency 传递
  SAVE_TEMPS 可选，非空时追加 --save-temps
  XO_OUTPUT  可选，自定义中间 .xo 路径（默认根据输出名生成）

位置参数：
  1          最终 .xclbin 输出文件名（默认 double_sha256.awsxclbin）
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ -z "${PLATFORM:-}" ]]; then
    echo "[build.sh] 错误: 必须设置 PLATFORM 环境变量" >&2
    usage
    exit 1
fi

VPP_BIN=${VPP:-v++}
TARGET_MODE=${TARGET:-hw}
KERNEL_SRC=${KERNEL:-fpga/kernels/double_sha256.cl}
OUTPUT=${1:-double_sha256.awsxclbin}
XO_OUTPUT=${XO_OUTPUT:-${OUTPUT%.awsxclbin}.xo}

declare -a FREQ_FLAG=()
if [[ -n "${FREQ_MHZ:-}" ]]; then
    FREQ_FLAG=("--kernel_frequency" "${FREQ_MHZ}")
fi

declare -a SAVE_FLAG=()
if [[ -n "${SAVE_TEMPS:-}" ]]; then
    SAVE_FLAG=("--save-temps")
fi

# 编译阶段：CL -> XO
COMPILE_CMD=("${VPP_BIN}" \
    -c \
    -t "${TARGET_MODE}" \
    --platform "${PLATFORM}" \
    -k double_sha256_fpga \
    -o "${XO_OUTPUT}" \
    "${KERNEL_SRC}")
if ((${#FREQ_FLAG[@]})); then
    COMPILE_CMD+=("${FREQ_FLAG[@]}")
fi
if ((${#SAVE_FLAG[@]})); then
    COMPILE_CMD+=("${SAVE_FLAG[@]}")
fi

# 链接阶段：XO -> XCLBIN
LINK_CMD=("${VPP_BIN}" \
    -l \
    -t "${TARGET_MODE}" \
    --platform "${PLATFORM}" \
    -o "${OUTPUT}" \
    "${XO_OUTPUT}")
if ((${#SAVE_FLAG[@]})); then
    LINK_CMD+=("${SAVE_FLAG[@]}")
fi

set -x
"${COMPILE_CMD[@]}"
"${LINK_CMD[@]}"
set +x

echo "[build.sh] 编译完成: ${OUTPUT}"
