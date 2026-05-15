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
├── gemma4_31b.Dockerfile       # base + DFlash overlay (PR #41703 + #40391 + club-3090 #42102, 17 files)
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
- **Gemma 4 31B** (TP=2, INT8 per-token-head target KV, z-lab DFlash drafter n=7 in independent BF16 KV pool, 65K-262K ctx, vision + tools) — extracted from [`gemma-4-31b/vllm/compose/dual/dflash-int8.yml`](https://github.com/noonghunna/club-3090/blob/57eb269/models/gemma-4-31b/vllm/compose/dual/dflash-int8.yml). 17 patched files baked in via multi-stage build, covering three stacked PRs:
    - [vLLM PR #41703](https://github.com/vllm-project/vllm/pull/41703) — z-lab DFlash drafter (12 files): adds the DFlash spec-decode method + `qwen3_dflash` model registration so the `gemma-4-31B-it-DFlash` drafter loads.
    - [vLLM PR #40391 (rebased)](https://github.com/vllm-project/vllm/pull/40391) — pads Gemma 4's global-layer KV spec to a 1040-byte factor so per-token-head KV's page sizes can be unified across the two head_dims (256 sliding / 512 full).
    - [vLLM PR #42102](https://github.com/vllm-project/vllm/pull/42102) — club-3090's own three-layer fix that unblocks DFlash + INT8 PTH coexistence: (a) partition DFlash drafter's BF16 KV into an independent KV group before page-size unify, (b) override drafter `cache_dtype` to `"auto"` when engine global is quantized, (c) have the FA metadata scheduler read per-spec dtype when `kv_quant_mode == NONE`.
  - Earlier we shipped the lighter `int8.yml` extraction (Google MTP drafter + PR #40391 only). It degenerated into multilingual garbage on the 4th-7th turn of any chat with short replies — a logit-collapse signature consistent with KV-scale corruption that PR #42102 specifically fixes. club-3090's own README acknowledges the same class of bug in its "Paris smoke clean (no garbled output unlike the 2026-05-06 wrong-fix attempt)" line.
  - Context overrides exposed via env vars: `MAX_NUM_SEQS=2 + MAX_MODEL_LEN=65536` (default, code-optimal multi-tenant), `MAX_NUM_SEQS=1 + MAX_MODEL_LEN=262144` (single-stream long-ctx; effective KV pool ~168K tokens — requests larger than that reject).
  - Image base is `e47c98ef7a38...` (2026-05-06) — back to the `vllm-nightly-dflash` engine pin because the DFlash overlay was rebased onto that nightly. `transformers==5.8.0` is pip-installed at build time since that nightly ships 5.7.0 and DFlash's model registration requires 5.8.
