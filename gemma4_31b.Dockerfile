# Reproduces club-3090's models/gemma-4-31b/vllm/compose/dual/dflash-int8.yml
# (Gemma 4 31B AutoRound INT4 + z-lab DFlash drafter n=7, TP=2, INT8 PTH
# target KV + independent BF16 drafter KV pool) without host pollution.
#
# Previous incarnation used the lighter int8.yml overlay (Google MTP +
# PR #40391 only). That setup repeatably degenerated into multilingual
# garbage on the 4-7th turn of any chat with short replies — a logit-
# collapse signature consistent with KV scale corruption on the rebased
# PR #40391 path. club-3090's own dflash-int8 README acknowledges the
# same class of bug in its "Paris smoke clean (no garbled output unlike
# the 2026-05-06 wrong-fix attempt)" line, and ships PR #42102 to fix
# the underlying spec-decode + INT8 PTH coexistence issue.
#
# Overlay stack (17 files baked via multi-stage build):
#
#   1. vLLM PR #41703 (z-lab DFlash drafter, 12 files) — adds the
#      DFlash spec-decode method + qwen3_dflash model registration so
#      the gemma-4-31B-it-DFlash drafter is loadable.
#   2. vLLM PR #40391 rebased (per-token-head KV, 5 files merged with
#      #41703) — same intent as the lighter overlay but anchored to
#      the e47c98ef base + with subsequent fixes.
#   3. vLLM PR #42102 (club-3090's own fix, 3 of the 17 files modified)
#      — partitions DFlash drafter's BF16 KV spec into an independent
#      KV group before page-size unify, overrides drafter cache_dtype
#      to "auto" when engine global is quantized, makes the FA metadata
#      scheduler read per-spec dtype when kv_quant_mode is NONE. This
#      is what unblocks DFlash + INT8 PTH coexistence on Ampere.
#
# Base image bumped back to `e47c98ef` (2026-05-06) because the overlay
# was rebased onto that nightly. Engine-profile in club-3090 is
# `vllm-nightly-dflash`. transformers 5.8.0 is required by the DFlash
# drafter's model registration and is pip-installed at build time
# (the e47c98ef nightly ships 5.7.0).

FROM alpine/git:latest AS overlays
ARG CLUB3090_REF=57eb269cd70935fc3069b85e46ead8f0f0af13dc
RUN git clone https://github.com/noonghunna/club-3090.git /club-3090 \
 && git -C /club-3090 checkout ${CLUB3090_REF}

FROM vllm/vllm-openai:nightly-e47c98ef7a38792996e452ef53914e21e41928e9

RUN pip install --quiet --upgrade transformers==5.8.0

ARG SRC=/club-3090/models/gemma-4-31b/vllm/patches/vllm-gemma4-dflash-int8
ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm

# --- config layer (DFlash) ---
COPY --from=overlays ${SRC}/config/attention.py        ${SITE}/config/attention.py
COPY --from=overlays ${SRC}/config/speculative.py      ${SITE}/config/speculative.py

# --- model_executor layer ---
COPY --from=overlays ${SRC}/model_executor/layers/attention/attention.py ${SITE}/model_executor/layers/attention/attention.py
COPY --from=overlays ${SRC}/model_executor/models/gemma4.py              ${SITE}/model_executor/models/gemma4.py
COPY --from=overlays ${SRC}/model_executor/models/qwen3_dflash.py        ${SITE}/model_executor/models/qwen3_dflash.py

# --- transformers_utils (DFlash model registration) ---
COPY --from=overlays ${SRC}/transformers_utils/configs/speculators/algos.py ${SITE}/transformers_utils/configs/speculators/algos.py

# --- v1 attention (DFlash backend wiring) ---
COPY --from=overlays ${SRC}/v1/attention/backends/triton_attn.py ${SITE}/v1/attention/backends/triton_attn.py
COPY --from=overlays ${SRC}/v1/attention/backends/flash_attn.py  ${SITE}/v1/attention/backends/flash_attn.py
COPY --from=overlays ${SRC}/v1/attention/selector.py             ${SITE}/v1/attention/selector.py

# --- v1 core (DFlash + PR #40391 merged) ---
COPY --from=overlays ${SRC}/v1/core/kv_cache_utils.py     ${SITE}/v1/core/kv_cache_utils.py
COPY --from=overlays ${SRC}/v1/core/sched/scheduler.py    ${SITE}/v1/core/sched/scheduler.py

# --- v1 spec_decode (DFlash) ---
COPY --from=overlays ${SRC}/v1/spec_decode/dflash.py ${SITE}/v1/spec_decode/dflash.py
COPY --from=overlays ${SRC}/v1/spec_decode/eagle.py  ${SITE}/v1/spec_decode/eagle.py
COPY --from=overlays ${SRC}/v1/spec_decode/utils.py  ${SITE}/v1/spec_decode/utils.py

# --- v1 kv_cache_interface (PR #40391) ---
COPY --from=overlays ${SRC}/v1/kv_cache_interface.py ${SITE}/v1/kv_cache_interface.py

# --- v1 worker (DFlash + PR #40391 merged) ---
COPY --from=overlays ${SRC}/v1/worker/gpu/attn_utils.py        ${SITE}/v1/worker/gpu/attn_utils.py
COPY --from=overlays ${SRC}/v1/worker/gpu_model_runner.py      ${SITE}/v1/worker/gpu_model_runner.py
COPY --from=overlays ${SRC}/v1/worker/kv_cache_shape_utils.py  ${SITE}/v1/worker/kv_cache_shape_utils.py
