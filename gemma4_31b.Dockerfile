# Reproduces club-3090's models/gemma-4-31b/vllm/compose/dual/int8.yml
# (Gemma 4 31B AutoRound INT4 + Google MTP drafter n=4, TP=2, INT8
# per-token-head KV, 262K-capable ctx) without host pollution.
#
# The simpler dual/docker-compose.yml ships a 32K BF16 KV path that needs
# no overlays. We instead extract the int8.yml path because it unlocks
# 262K context on Ampere — same surface as the Qwen extraction. Three
# overlays are fetched in-builder and COPY'd into the image.
#
#   1. vLLM PR #40391 (rebased) — Gemma 4 per-token-head KV. Gemma 4's
#      interleaved attention has two head_dims (256 sliding / 512 full),
#      which breaks vLLM's KV page-size unification for any per-token-head
#      KV format. The PR pads the global layers' KV spec to a 1040-byte
#      factor; the rebase resolves the Mamba-hybrid attention conflict.
#      Drop when PR lands upstream (currently open + stalled on review).
#   2. vLLM PR #42006 — gemma4 tool parser, MTP streaming multi-tool
#      calls (empty tool_calls[] when MTP bundles last param + closing
#      </function> in one delta).
#   3. vLLM PR #41991 — gemma4 tool parser, infinite-loop + array-bounds
#      fixes in the parser helpers.
# #42006 and #41991 are stacked into one file by club-3090.
#
# Why this image SHA: the rebase was performed against post-Mamba-hybrid
# main on 2026-05-08, which lines up with the `1acd67a795...` nightly
# (May 8) — NOT the older `e47c98ef...` (May 6) that the
# `vllm-nightly-full` profile yaml pins. The calibration record in
# club-3090 (`scripts/lib/profiles/calibration/gemma-4-31b.yml`)
# benches int8.yml against `vllm-nightly-1acd67a79`. The rebased
# `attn_utils.py` also imports `ModelSpecificAttnMetadata` from
# `vllm.v1.worker.gpu.model_states.interface`, which doesn't exist in
# `e47c98ef` (verified upstream).
#
# Same image as Qwen — both models share one nightly. fp8_e5m2 still
# hits a gemma4_mm.py allowlist assert and fp8_e4m3 still needs
# `fp8e4nv` (Hopper+); INT8 PTH dispatches via standard PyTorch
# torch.int8 ops, so it runs on Ampere SM 8.6.

FROM alpine/git:latest AS overlays
ARG CLUB3090_REF=57eb269cd70935fc3069b85e46ead8f0f0af13dc
RUN git clone https://github.com/noonghunna/club-3090.git /club-3090 \
 && git -C /club-3090 checkout ${CLUB3090_REF}

FROM vllm/vllm-openai:nightly-1acd67a795ebccdf9b9db7697ae9082058301657

ARG PR40391=/club-3090/models/gemma-4-31b/vllm/patches/vllm-pr40391-rebased
ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

# --- PR #40391 rebased (7 files) ---
COPY --from=overlays ${PR40391}/model_executor/layers/attention/attention.py ${SITE}/model_executor/layers/attention/attention.py
COPY --from=overlays ${PR40391}/model_executor/models/gemma4.py ${SITE}/model_executor/models/gemma4.py
COPY --from=overlays ${PR40391}/v1/core/kv_cache_utils.py ${SITE}/v1/core/kv_cache_utils.py
COPY --from=overlays ${PR40391}/v1/kv_cache_interface.py ${SITE}/v1/kv_cache_interface.py
COPY --from=overlays ${PR40391}/v1/worker/gpu/attn_utils.py ${SITE}/v1/worker/gpu/attn_utils.py
COPY --from=overlays ${PR40391}/v1/worker/gpu_model_runner.py ${SITE}/v1/worker/gpu_model_runner.py
COPY --from=overlays ${PR40391}/v1/worker/kv_cache_shape_utils.py ${SITE}/v1/worker/kv_cache_shape_utils.py

# --- PR #42006 + #41991 stacked (1 file) ---
COPY --from=overlays \
    /club-3090/models/gemma-4-31b/vllm/patches/vllm-gemma4-tool-parser-fixes/tool_parsers/gemma4_tool_parser.py \
    ${SITE}/tool_parsers/gemma4_tool_parser.py
