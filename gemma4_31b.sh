#!/bin/sh
# Usage:
#   ./gemma4_31b.sh --build         # build the image
#   ./gemma4_31b.sh [PORT]          # run; PORT defaults to 8030
#
# vLLM auto-downloads Intel/gemma-4-31B-it-int4-AutoRound (~21 GB) and the
# Google MTP drafter google/gemma-4-31B-it-assistant (~0.5B / 0.9 GB BF16)
# into the mounted HF cache on first boot.
#
# Engine config mirrors club-3090/models/gemma-4-31b/vllm/compose/dual/int8.yml
# (TP=2, INT8 per-token-head KV via vendored PR #40391, MTP n=4, 98K ctx
# default, vision + tools).
#
# Context vs. concurrency trade (KV pool is shared across streams):
#   MAX_NUM_SEQS=4 + MAX_MODEL_LEN=98304   (default — multi-tenant agent)
#   MAX_NUM_SEQS=2 + MAX_MODEL_LEN=174080  (2 concurrent long-ctx agents)
#   MAX_NUM_SEQS=1 + MAX_MODEL_LEN=262144  (single-stream, model native max)
#
# KV format notes (do NOT change without re-reading gemma4_31b.Dockerfile):
#   - int8_per_token_head is the only PTH variant that runs on Ampere SM 8.6.
#     fp8_e5m2 hits a Gemma 4 allowlist assert; fp8_e4m3 needs Triton fp8e4nv
#     which Ampere doesn't implement. Override KV_DTYPE=fp8_per_token_head
#     only on Ada/Blackwell.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="gemma4_31b"
IMAGE="vllm_${NAME}:latest"
CONTAINER="vllm_${NAME}"

if [ "$1" = "--build" ] || [ "$1" = "-b" ]; then
	exec docker build -f "${HERE}/${NAME}.Dockerfile" -t "${IMAGE}" "${HERE}"
fi

if [ "$1" = "--stop" ]; then
	exec docker stop "${CONTAINER}"
fi

PORT="${1:-8030}"
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
	-e OMP_NUM_THREADS=1 \
	-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
	-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
	"${IMAGE}" \
	--model Intel/gemma-4-31B-it-int4-AutoRound \
	--served-model-name gemma-4-31b-autoround \
	--tensor-parallel-size 2 \
	--disable-custom-all-reduce \
	--max-model-len "${MAX_MODEL_LEN:-98304}" \
	--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
	--max-num-seqs "${MAX_NUM_SEQS:-4}" \
	--max-num-batched-tokens 4096 \
	--kv-cache-dtype "${KV_DTYPE:-int8_per_token_head}" \
	--trust-remote-code \
	--enable-auto-tool-choice \
	--tool-call-parser gemma4 \
	--chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja \
	--speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}' \
	--host 0.0.0.0 \
	--port 8000
