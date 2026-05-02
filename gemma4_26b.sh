#!/bin/sh
# Usage:
#   ./gemma4_26b.sh --build         # build the image
#   ./gemma4_26b.sh [PORT]          # run; PORT defaults to 8000
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="gemma4_26b"
IMAGE="vllm_${NAME}:latest"

if [ "$1" = "--build" ] || [ "$1" = "-b" ]; then
	exec docker build -f "${HERE}/${NAME}.Dockerfile" -t "${IMAGE}" "${HERE}"
fi

PORT="${1:-8000}"
CONTAINER="vllm_${NAME}_$$"
trap 'docker stop "${CONTAINER}" >/dev/null 2>&1 || true' EXIT INT TERM
mkdir -p "${HERE}/cache/${NAME}/torch_compile" "${HERE}/cache/${NAME}/triton"
docker run --rm --name "${CONTAINER}" --device nvidia.com/gpu=all \
	-v "${HERE}/huggingface:/root/.cache/huggingface" \
	-v "${HERE}/cache/${NAME}/torch_compile:/root/.cache/vllm/torch_compile_cache" \
	-v "${HERE}/cache/${NAME}/triton:/root/.triton/cache" \
	-p ${PORT}:8000 \
	--ipc=host \
	"${IMAGE}" \
	cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
	--tensor-parallel-size 2 \
	--max-model-len 32768 \
	--limit-mm-per-prompt '{"image":0,"audio":0}' \
	--enable-prefix-caching \
	--max-num-batched-tokens 4096 \
	--gpu-memory-utilization 0.9 \
	--max-num-seqs 128 \
	--safetensors-load-strategy lazy \
	--default-chat-template-kwargs '{"enable_thinking": false}'
