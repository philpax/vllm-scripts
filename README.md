# vllm-scripts

A small collection of self-contained shell scripts and Dockerfiles for running [vLLM](https://github.com/vllm-project/vllm) in Docker on a dual RTX 3090 box. Each model has one `.sh` (driver) and one `.Dockerfile` (custom image with any required patches baked in).

Design goals:

- **No host pollution.** Patches live inside the image, not in `/opt/...` mounts. The only host paths are `huggingface/` (model weights) and `cache/<name>/` (torch.compile + Triton kernel caches), both inside this directory.
- **Drop-in.** Build once via `./<model>.sh --build`, then run the same script to start the server. vLLM auto-downloads weights from HF Hub on first boot.
- **Configurable port.** Every script takes the port as its first positional arg; see each script for its default.

## Layout

```
vllm/
├── README.md
├── .gitignore
├── gemma4_26b.Dockerfile       # base + transformers upgrade
├── gemma4_26b.sh               # PORT defaults to 8000
├── qwen36_27b.Dockerfile       # base + Marlin pad-sub-tile-n (vLLM PR #40361)
├── qwen36_27b.sh               # PORT defaults to 8010
├── huggingface/                # shared HF cache (gitignored)
└── cache/
    ├── gemma4_26b/{torch_compile,triton}/   # gitignored, created on first run
    └── qwen36_27b/{torch_compile,triton}/
```

## Usage

Each script is both the builder and the runner:

```sh
# Build the image for the model you want (one-time, or after editing the Dockerfile)
./qwen36_27b.sh --build

# Run on the default port (8010 for qwen36_27b, 8000 for gemma4_26b)
./qwen36_27b.sh

# Run on a custom port (first positional arg)
./qwen36_27b.sh 9000
```

Same pattern for `gemma4_26b.sh`.

## Credits

The Qwen 3.6 27B configuration (TP=2, fp8_e5m2 KV, MTP n=3, 262K ctx, vision + tools, plus the Marlin pad-sub-tile-n patch from [vLLM PR #40361](https://github.com/vllm-project/vllm/pull/40361)) is a simplified extraction of [`docker-compose.dual.yml`](https://github.com/noonghunna/club-3090/blob/e0e1752b3a0299671a4c5fec610fa3e321caed95/models/qwen3.6-27b/vllm/compose/docker-compose.dual.yml) from [noonghunna/club-3090](https://github.com/noonghunna/club-3090) at commit `e0e1752b3a0299671a4c5fec610fa3e321caed95`. See that repo for the full multi-variant setup, benchmarks, and the engineering rationale behind each flag.
