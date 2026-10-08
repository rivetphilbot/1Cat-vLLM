#!/bin/bash
# Build libexl3_sm70.so (SM70 EXL3 MoE kernels) next to exl3_sm70_ops.py, where Exl3ExpertBank loads it via ctypes.
# Override the location at runtime with EXL3_SM70_LIB=/path/to/libexl3_sm70.so.
# Not wired into CMake yet: the kernels load through ctypes rather than a torch extension (see docs/design/exl3_sm70.md).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NVCC=${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}
CCBIN=${NVCC_CCBIN:+-ccbin $NVCC_CCBIN}
$NVCC -O3 -arch=sm_70 -std=c++17 -shared -Xcompiler -fPIC $CCBIN \
  -o "$ROOT/vllm/model_executor/layers/quantization/libexl3_sm70.so" \
  "$ROOT/csrc/quantization/exl3_sm70/exl3_sm70.cu"
echo "built $ROOT/vllm/model_executor/layers/quantization/libexl3_sm70.so"
