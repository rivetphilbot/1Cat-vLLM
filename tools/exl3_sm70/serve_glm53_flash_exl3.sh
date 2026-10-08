#!/bin/bash
# Example: GLM-5.3-Flash EXL3 (2-bit routed experts, sm70_colmajor) on 4x V100-32GB (NVLink), TP4.
# MODEL = a checkpoint produced by repack_1cat.py (add --mtp there to keep the MTP head for SPEC=mtp).
# Measured on 4x V100-32GB: 163,840 tokens of context at GPU_UTIL 0.92 with vision and speculation on.
set -euo pipefail
: "${MODEL:?set MODEL to the repacked checkpoint directory}"
SPEC=${SPEC:-mtp}
if [ "$SPEC" = mtp ]; then
  export VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=${VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT:-0}
  SPEC_CFG={\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"greedy\"}
else
  : "${DRAFT:?set DRAFT to a DFlash2 draft model directory}"
  SPEC_CFG={\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":4}
fi
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name glm-5.3-flash \
  --trust-remote-code --dtype half \
  --tensor-parallel-size ${TP:-4} --kv-cache-dtype ${KV_DTYPE:-fp8_e4m3} \
  --max-model-len ${MAX_MODEL_LEN:-163840} --max-num-seqs ${MAX_NUM_SEQS:-1} \
  --max-num-batched-tokens ${MAX_BATCHED:-4096} --gpu-memory-utilization ${GPU_UTIL:-0.92} \
  --reasoning-parser deepseek_r1 --tool-call-parser glm47 --enable-auto-tool-choice \
  --speculative-config "$SPEC_CFG" "$@"
