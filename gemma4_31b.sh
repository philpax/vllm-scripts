#!/bin/sh
# Usage:
#   ./gemma4_31b.sh --build         # build the image
#   ./gemma4_31b.sh [PORT]          # run; PORT defaults to 8030
#
# vLLM auto-downloads Intel/gemma-4-31B-it-int4-AutoRound (~21 GB) and the
# z-lab DFlash drafter z-lab/gemma-4-31B-it-dflash (~2.9 GB BF16) into the
# mounted HF cache on first boot.
#
# Engine config mirrors club-3090/models/gemma-4-31b/vllm/compose/dual/dflash-int8.yml
# (TP=2, INT8 PTH target KV via PR #40391 rebased + drafter BF16 KV pool
# via PR #42102, DFlash n=7, vision + tools).
#
# Context vs. concurrency trade — INT8 PTH expands KV pool but DFlash
# drafter footprint (2.9 GB BF16, larger than the MTP assistant's 0.5 GB)
# narrows the headroom relative to the int8.yml path:
#   MAX_NUM_SEQS=2 + MAX_MODEL_LEN=65536   (default — code-optimal multi-tenant)
#   MAX_NUM_SEQS=1 + MAX_MODEL_LEN=131072  (single-stream long-ctx)
#   MAX_NUM_SEQS=1 + MAX_MODEL_LEN=262144  (model native max; ~168K effective
#                                            KV pool — requests >168K reject)
#
# Why DFlash over Google MTP: the lighter int8.yml extraction we had before
# degenerated into multilingual garbage on the 4-7th turn of any chat with
# short replies. Same KV format (INT8 PTH), same target weights — only the
# drafter + the surrounding overlay set differs. DFlash carries club-3090's
# PR #42102 fix for the spec-decode + INT8 PTH coexistence bug that the
# raw PR #40391 rebase doesn't.
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
	--dtype bfloat16 \
	--tensor-parallel-size 2 \
	--disable-custom-all-reduce \
	--max-model-len "${MAX_MODEL_LEN:-65536}" \
	--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
	--max-num-seqs "${MAX_NUM_SEQS:-2}" \
	--max-num-batched-tokens 4096 \
	--kv-cache-dtype "${KV_DTYPE:-int8_per_token_head}" \
	--trust-remote-code \
	--enable-auto-tool-choice \
	--tool-call-parser gemma4 \
	--chat-template /vllm-workspace/examples/tool_chat_template_gemma4.jinja \
	--speculative-config '{"method":"dflash","model":"z-lab/gemma-4-31B-it-dflash","num_speculative_tokens":7}' \
	--host 0.0.0.0 \
	--port 8000
