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
  FREQ_MHZ   可选，期望频率（MHz），通过 --kernel_frequency 传递，默认空
  SAVE_TEMPS 可选，非空时追加 --save-temps

位置参数：
  1          输出文件名（默认 double_sha256.awsxclbin）
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
FREQ_FLAG=()
if [[ -n "${FREQ_MHZ:-}" ]]; then
    FREQ_FLAG=("--kernel_frequency" "${FREQ_MHZ}")
fi
SAVE_FLAG=()
if [[ -n "${SAVE_TEMPS:-}" ]]; then
    SAVE_FLAG=("--save-temps")
fi

set -x
"${VPP_BIN}" \
    -t "${TARGET_MODE}" \
    --platform "${PLATFORM}" \
    -k double_sha256_fpga \
    -o "${OUTPUT}" \
    "${FREQ_FLAG[@]}" \
    "${SAVE_FLAG[@]}" \
    "${KERNEL_SRC}"
set +x

echo "[build.sh] 编译完成: ${OUTPUT}"
