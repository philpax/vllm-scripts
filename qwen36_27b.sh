#!/bin/sh
# Usage:
#   ./qwen36_27b.sh --build         # build the image (clones the Marlin patch fork)
#   ./qwen36_27b.sh [PORT]          # run; PORT defaults to 8010
#
# vLLM auto-downloads Lorbus/Qwen3.6-27B-int4-AutoRound into the mounted HF cache
# on first boot (~16 GB).
#
# Engine config mirrors club-3090/models/qwen3.6-27b/vllm/compose/docker-compose.dual.yml
# (TP=2, fp8_e5m2 KV, MTP n=3, 262K ctx, vision + tools).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="qwen36_27b"
IMAGE="vllm_${NAME}_marlin:latest"

if [ "$1" = "--build" ] || [ "$1" = "-b" ]; then
	exec docker build -f "${HERE}/${NAME}.Dockerfile" -t "${IMAGE}" "${HERE}"
fi

PORT="${1:-8010}"
CONTAINER="vllm_${NAME}_$$"
trap 'docker stop "${CONTAINER}" >/dev/null 2>&1 || true' EXIT INT TERM
mkdir -p "${HERE}/cache/${NAME}/torch_compile" "${HERE}/cache/${NAME}/triton"
docker run --rm --name "${CONTAINER}" --device nvidia.com/gpu=all \
	-v "${HERE}/huggingface:/root/.cache/huggingface" \
	-v "${HERE}/cache/${NAME}/torch_compile:/root/.cache/vllm/torch_compile_cache" \
	-v "${HERE}/cache/${NAME}/triton:/root/.triton/cache" \
	-p ${PORT}:8000 \
	--ipc=host \
	--shm-size=16g \
	-e VLLM_WORKER_MULTIPROC_METHOD=spawn \
	-e NCCL_CUMEM_ENABLE=0 \
	-e NCCL_P2P_DISABLE=1 \
	-e VLLM_NO_USAGE_STATS=1 \
	-e VLLM_USE_FLASHINFER_SAMPLER=1 \
	-e OMP_NUM_THREADS=1 \
	-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
	"${IMAGE}" \
	--model Lorbus/Qwen3.6-27B-int4-AutoRound \
	--served-model-name qwen3.6-27b-autoround \
	--quantization auto_round \
	--dtype float16 \
	--tensor-parallel-size 2 \
	--disable-custom-all-reduce \
	--max-model-len 262144 \
	--gpu-memory-utilization 0.92 \
	--max-num-seqs 2 \
	--max-num-batched-tokens 8192 \
	--kv-cache-dtype fp8_e5m2 \
	--trust-remote-code \
	--reasoning-parser qwen3 \
	--enable-auto-tool-choice \
	--tool-call-parser qwen3_coder \
	--enable-prefix-caching \
	--enable-chunked-prefill \
	--safetensors-load-strategy lazy \
	--speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
	--host 0.0.0.0 \
	--port 8000
