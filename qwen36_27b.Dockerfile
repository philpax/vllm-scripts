# Reproduces club-3090's docker-compose.dual.yml (Qwen3.6-27B, dual 3090, fp8 KV,
# MTP n=3, vision/tools) without host pollution.
#
# Bakes vLLM PR #40361 (Marlin pad-sub-tile-n) into the image instead of
# volume-mounting from /opt/ai/vllm-src. The patch is required for TP=2 on
# AutoRound W4A16 quants where per-rank out-dim shards fall below Marlin's
# 64-thread minimum on Ampere SM 8.6. Drop both COPYs once the PR lands
# upstream.

FROM alpine/git:latest AS marlin-src
ARG MARLIN_REF=marlin-pad-sub-tile-n
RUN git clone --depth 1 -b ${MARLIN_REF} \
    https://github.com/noonghunna/vllm.git /vllm-src

FROM vllm/vllm-openai:nightly-01d4d1ad375dc5854779c593eee093bcebb0cada

COPY --from=marlin-src \
    /vllm-src/vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py
COPY --from=marlin-src \
    /vllm-src/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py
