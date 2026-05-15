# Reproduces club-3090's models/qwen3.6-27b/vllm/compose/dual/docker-compose.yml
# (Qwen3.6-27B, dual 3090, fp8 KV, MTP n=3, vision/tools) without host
# pollution. Both overlays are fetched in-builder and COPY'd into the image.
#
#   1. vLLM PR #40361 — Marlin pad-sub-tile-n. Required for TP=2 on
#      AutoRound W4A16 where per-rank out-dim shards fall below Marlin's
#      64-thread minimum on Ampere SM 8.6. Drop when PR lands upstream.
#   2. vLLM PR #35936 — required-tool fallback. Without it, tool_choice=
#      "required" + qwen3_coder parser returns empty tool_calls[] because
#      vLLM's required path emits JSON while the configured parser scans
#      for XML <tool_call> sentinels. Drop when PR lands upstream.
#   3. froggeric/Qwen-Fixed-Chat-Templates — patched chat_template.jinja
#      that fixes 7 default-template bugs (empty <think></think> spam,
#      </thinking> hallucination, unclosed think before tool call, no-
#      user-query crash, developer role, etc.). Pairs with the
#      --default-chat-template-kwargs '{"enable_thinking": false}' flag in
#      the script.
#
# club-3090's compose mounts the PR #35936 files at side paths and runs a
# sidecar install.sh at boot, but that pattern only exists to coexist with
# Genesis (which RW-patches the same files at vllm import). Genesis is not
# part of this extraction, so we COPY at build time and skip the dance.

FROM alpine/git:latest AS overlays
ARG MARLIN_REF=marlin-pad-sub-tile-n
ARG CLUB3090_REF=57eb269cd70935fc3069b85e46ead8f0f0af13dc
RUN git clone --depth 1 -b ${MARLIN_REF} \
        https://github.com/noonghunna/vllm.git /vllm-marlin \
 && git clone https://github.com/noonghunna/club-3090.git /club-3090 \
 && git -C /club-3090 checkout ${CLUB3090_REF}

FROM vllm/vllm-openai:nightly-1acd67a795ebccdf9b9db7697ae9082058301657

# --- PR #40361 (Marlin pad-sub-tile-n) ---
COPY --from=overlays \
    /vllm-marlin/vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/marlin.py
COPY --from=overlays \
    /vllm-marlin/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py

# --- PR #35936 (required-tool fallback) ---
# chat_completion/serving.py is byte-identical to the pinned upstream nightly
# today (PR #35936's streaming hunks don't apply on this pin); we COPY it
# anyway as a future-ready slot for when the streaming hunks land.
COPY --from=overlays \
    /club-3090/models/qwen3.6-27b/vllm/patches/vllm-pr35936-required-fallback/vllm/entrypoints/openai/chat_completion/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
COPY --from=overlays \
    /club-3090/models/qwen3.6-27b/vllm/patches/vllm-pr35936-required-fallback/vllm/entrypoints/openai/engine/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/serving.py

# --- froggeric Qwen-Fixed-Chat-Templates ---
COPY --from=overlays \
    /club-3090/models/qwen3.6-27b/vllm/patches/froggeric-chat-template/chat_template.jinja \
    /etc/qwen-froggeric-chat-template.jinja
