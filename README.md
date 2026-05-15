# vllm-scripts

A small collection of self-contained shell scripts and Dockerfiles for running [vLLM](https://github.com/vllm-project/vllm) in Docker on a dual RTX 3090 box. Each model has one `.sh` (driver) and one `.Dockerfile` (custom image with any required patches baked in).

Design goals:

- **No host pollution.** Patches are fetched inside multi-stage Docker builds (via `git clone`) and `COPY`'d into the final image. The only host paths are `huggingface/` (model weights) and `cache/<name>/` (torch.compile + Triton kernel caches), both inside this directory.
- **Drop-in.** Build once via `./<model>.sh --build`, then run the same script to start the server. vLLM auto-downloads weights from HF Hub on first boot.
- **Configurable port.** Every script takes the port as its first positional arg; see each script for its default.

## Layout

```
vllm/
├── README.md
├── .gitignore
├── gemma4_26b.Dockerfile       # base + transformers upgrade
├── gemma4_26b.sh               # PORT defaults to 8000
├── gemma4_31b.Dockerfile       # base + PR #40391 (int8 PTH KV) + #42006/#41991 (tool parser)
├── gemma4_31b.sh               # PORT defaults to 8030
├── qwen36_27b.Dockerfile       # base + PR #40361 (Marlin pad) + #35936 (required-tool fallback) + froggeric template
├── qwen36_27b.sh               # PORT defaults to 8010
├── huggingface/                # shared HF cache (gitignored)
└── cache/
    ├── gemma4_26b/{torch_compile,triton}/   # gitignored, created on first run
    ├── gemma4_31b/{torch_compile,triton}/
    └── qwen36_27b/{torch_compile,triton}/
```

## Usage

Each script is both the builder and the runner:

```sh
# Build the image for the model you want (one-time, or after editing the Dockerfile)
./qwen36_27b.sh --build

# Run on the default port (8010 for qwen36_27b, 8000 for gemma4_26b, 8030 for gemma4_31b)
./qwen36_27b.sh

# Run on a custom port (first positional arg)
./qwen36_27b.sh 9000
```

Same pattern for `gemma4_26b.sh` and `gemma4_31b.sh`.

## Credits

Both Qwen and Gemma configs are simplified extractions from [noonghunna/club-3090](https://github.com/noonghunna/club-3090) at commit `57eb269`. See that repo for the full multi-variant setup, benchmarks, and the engineering rationale behind each flag.

- **Qwen 3.6 27B** (TP=2, fp8_e5m2 KV, MTP n=3, 262K ctx, vision + tools) — extracted from [`qwen3.6-27b/vllm/compose/dual/docker-compose.yml`](https://github.com/noonghunna/club-3090/blob/57eb269/models/qwen3.6-27b/vllm/compose/dual/docker-compose.yml). Three overlays are baked in via multi-stage Docker build:
    - [vLLM PR #40361](https://github.com/vllm-project/vllm/pull/40361) — Marlin pad-sub-tile-n (required for TP=2 AutoRound INT4 where per-rank out-dim shards fall below Marlin's 64-thread minimum on Ampere SM 8.6).
    - [vLLM PR #35936](https://github.com/vllm-project/vllm/pull/35936) — required-tool fallback (without it, `tool_choice: "required"` + the `qwen3_coder` parser returns empty `tool_calls[]`).
    - [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — patched `chat_template.jinja` baked at `/etc/qwen-froggeric-chat-template.jinja` and wired via `--chat-template`. Fixes 7 default-template bugs (empty `<think></think>` spam, `</thinking>` hallucination, unclosed think before tool call, etc.). Pairs with `--default-chat-template-kwargs '{"enable_thinking": false}'`.
  - Skipped from the canonical compose: NVLink auto-detect entrypoint (we're PCIe-only, so always `--disable-custom-all-reduce`).
- **Gemma 4 31B** (TP=2, INT8 per-token-head KV, Google MTP "assistant" drafter n=4, 98K-262K ctx, vision + tools) — extracted from [`gemma-4-31b/vllm/compose/dual/int8.yml`](https://github.com/noonghunna/club-3090/blob/57eb269/models/gemma-4-31b/vllm/compose/dual/int8.yml). The simpler `dual/docker-compose.yml` ships a 32K BF16 default that needs no overlays, but Gemma 4's two-head_dim attention forces *any* per-token-head KV format to break vLLM's page-size unification — so 262K on Ampere requires the rebased PR overlay. Three patches are baked in:
    - [vLLM PR #40391 (rebased)](https://github.com/vllm-project/vllm/pull/40391) — pads the global layers' KV spec to a 1040-byte factor so the allocator's slab structure can absorb the 256/512 head_dim mismatch; the local rebase resolves the post-Mamba-hybrid main conflict.
    - [vLLM PRs #42006](https://github.com/vllm-project/vllm/pull/42006) + [#41991](https://github.com/vllm-project/vllm/pull/41991) (stacked) — gemma4 tool-parser bug fixes: MTP streaming multi-tool calls + infinite-loop / array-bounds bugs in parser helpers.
  - Context vs. concurrency trade is exposed via env vars: `MAX_NUM_SEQS=4 + MAX_MODEL_LEN=98304` (default, multi-tenant), `MAX_NUM_SEQS=1 + MAX_MODEL_LEN=262144` (single-stream, model native max).
  - Image base is the same `1acd67a795...` nightly (2026-05-08) as Qwen. The `vllm-nightly-full` profile yaml in club-3090 nominally pins int8.yml to an older `e47c98ef` (2026-05-06), but the actual rebase imports `ModelSpecificAttnMetadata` from `model_states/interface.py` — a class that doesn't exist until `1acd67a795`. Verified empirically and against the `gemma-4-31b.yml` calibration record (int8 benched on `vllm-nightly-1acd67a79`).
