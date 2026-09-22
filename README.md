# DartAI Runtime — Ternary LLMs on llama.cpp, any GPU

Run a **27-billion-parameter model in 5.5–6.7 GB** — on NVIDIA (CUDA), AMD / Intel / NVIDIA (Vulkan) or CPU.
Prebuilt binaries for Linux and Windows; no compilation, no Python, no cloud. OpenAI-compatible API out of the box.

*Keywords: ternary LLM · 1.58-bit · 2-bit quantization · BitNet-style · low VRAM · 6 GB GPU · GTX 1060 ·
llama.cpp · GGUF · Vulkan · CUDA · AMD Strix Halo · Radeon 8060S · Qwen3.8-27B · Bonsai · Prism ML ·
local LLM · offline inference · OpenAI-compatible server · speculative decoding · kernel autotuning.*

## Tested hardware at a glance

Every number below was **measured** on the machine named, with the released binaries (Bonsai-2-27B, 2026-09-19..21,
mains power). "—" means not measured; "does not fit" means the model + draft do not fit in that memory.
Speculative decoding = DFlash2 Q4_K_M draft (`--spec-type draft-dflash`); its gain depends on the prompt
(code accepts ~50–70% of draft tokens, prose less).

| GPU | memory | backend | decode TQ2_1 | decode TQ1_1 | with speculative decoding | quality (Winogrande / HellaSwag / MMLU) |
|---|---|---|---|---|---|---|
| **NVIDIA RTX 5070 Ti Laptop** (Blackwell) | 12 GB VRAM | CUDA | **44.4 tok/s** | 27.7 tok/s | **111.7 tok/s** on code (TQ2_1, n_max 7, 52% accepted) | 73.2 ± 2.0 / 75.6% / **41.7 ± 1.3** (1548 tasks) |
| **AMD Radeon 8060S** (Ryzen AI MAX+ 395, Strix Halo, iGPU) | 128 GB unified RAM | Vulkan | 23.2 tok/s | 19.7 tok/s | prose **−13%**, code +6% — not worth it on an iGPU | parent ruler measured here: Q6_K_XL 75.2 / 82.3% / 42.0 |
| **AMD Radeon 890M** (Ryzen AI 9 HX 370, Strix Point, iGPU) | 16 GB shared RAM | Vulkan | 8.9 tok/s | 6.7 tok/s | — | — |
| **NVIDIA GTX 1060** (Pascal) | 6 GB VRAM | CUDA | does not fit | runs with 44/64 layers on GPU (`-ub 8`, see limitations) | does not fit | Winogrande **73.2 ± 2.0** — identical logits to the 5070 Ti |

Decode is `llama-bench tg128`, batch 1. On the 890M the numbers are from an earlier build (2026-09-18); the
autotuner keeps the default kernel there, so they are representative. Quality rows are the same tasks and seed on
every GPU (details in [Quality](#quality)); Winogrande on the 1060 reproduced the 5070 Ti score to four decimals,
which is the correctness check: Pascal and Blackwell kernels compute the same logits.

## What this is

**DartAI Runtime** is [llama.cpp](https://github.com/ggml-org/llama.cpp) — the most widely used engine for
running LLMs locally — with two new weight formats, **TQ1_1** (1.75 bits/weight) and **TQ2_1** (2.13
bits/weight), and the CPU, Vulkan and CUDA kernels that make them fast. You download a package, download a
ternary `.gguf` model, and run a server with an OpenAI-compatible API on your own machine.

**Who it is for**: anyone who wants a **27B model on an ordinary GPU** — a 6 GB GTX 1060, a laptop with an
integrated Radeon, an 8–12 GB RTX — where conventional formats do not fit. The same Qwen3.8-27B needs
15.6 GB in Q4_K_M and 10.1 GB in Q2_K; in **TQ1_1 it needs 5.5 GB**.

## What "ternary" means

A standard LLM stores each weight as a 16-bit number. Quantization shrinks that — 8, 4, 2 bits per weight —
trading precision for memory. **Ternary** is the practical extreme: each weight can only be **−1, 0 or +1**.
Three states carry log₂(3) ≈ 1.58 bits of information; with packing and per-block scales the DartAI formats
land at **1.75 bpw (TQ1_1)** and **2.13 bpw (TQ2_1)**.

Two things make this work in practice, and both are measured in this repository:

- **The model must be trained ternary, not converted afterwards.** A model post-quantized to 2 bits (IQ2_XXS,
  Q2_K) degrades; Prism ML's Bonsai 2 is *trained* with ternary weights and keeps the parent's quality — that is
  the checkpoint this runtime runs. It is also why TQ1_1 and TQ2_1 give exactly the same quality: they are two
  lossless encodings of the same three states.
- **The kernel must actually cash in the missing bytes.** LLM decode is memory-bandwidth-bound; reading 1.75
  bits instead of 16 per weight is what makes a 27B decode at 44 tok/s on a laptop RTX 5070 Ti. On
  latency-bound GPUs (integrated graphics) the cost of unpacking trits weighs more — which is why the runtime
  **measures your GPU** instead of assuming.

What this runtime is **not**: it is not a quantizer (you cannot convert your own models to ternary here —
that is training), and it has no GUI — for that there is *LlamaForge by DartAI*.

## Packages (v0.2.0)
| file | for | size |
|---|---|---|
| `dartai-linux-x64-vulkan-0.2.0.tar.gz` | Linux x86-64 (glibc ≥ 2.35), any GPU with a Vulkan 1.2+ driver (AMD, Intel, NVIDIA incl. GTX 10xx) and CPU | 32 MB |
| `dartai-linux-x64-cuda12-0.2.0.tar.gz` | Linux x86-64, NVIDIA GTX 10xx → RTX 50xx (sm 61–120), CUDA 12 runtime included · driver ≥ 525 | 849 MB |
| `dartai-windows-x64-vulkan-0.2.0.zip` | Windows 10/11 x64, any GPU with a Vulkan 1.2+ driver | 26 MB |
| `dartai-windows-x64-cuda12-0.2.0.zip` | Windows 10/11 x64, NVIDIA GTX 10xx → RTX 50xx, CUDA 12 DLLs included · driver ≥ 525 | 555 MB |

Verify hashes with `SHA256SUMS`. The Windows packages were built with MSVC and validated on a GTX 1060
(ternary kernel correctness and the autotuner).

## Models (Hugging Face, Apache 2.0 — not redistributed here)
| model | format | size | when |
|---|---|---|---|
| [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | **TQ2_1** | 6.7 GB | GPU ≥ 8 GB, iGPU / unified memory ≥ 12 GB — **the fastest** |
| Ternary-Bonsai-2-27B | **TQ1_1** | 5.5 GB | the smallest; 6 GB GPUs with partial offload (see limitations) |
| Ternary-Bonsai-1.7B | TQ2_1 | 0.46 GB | quick test / weak machines |

## Usage
**Linux**
```bash
tar xzf dartai-linux-x64-cuda12-0.2.0.tar.gz && cd dartai-linux-x64-cuda12
./dartai-run.sh ~/models/Ternary-Bonsai-2-27B-TQ2_1.gguf 8080 8192   # picks -ngl from free VRAM
# OpenAI-compatible API at http://127.0.0.1:8080/v1 ; web UI at http://127.0.0.1:8080
```
Direct binary: `LD_LIBRARY_PATH=. ./llama-server -m model.gguf -ngl 99 -c 8192 --port 8080`.

**Windows**: unzip and run `llama-server.exe -m model.gguf -ngl 99 -c 8192 --port 8080` from inside the
folder (the DLLs live next to it).

## What's new in 0.2.0
- **Measurement-based kernel autotuner** (CUDA and Vulkan): on the first inference the runtime times the
  kernel alternatives *on your GPU* and caches the choice in `~/.cache/dartai/` (Windows:
  `%USERPROFILE%\.cache\dartai\`). Measured: +8–12% decode and **+117% speculative decoding** on an RTX 5070 Ti,
  **+22%** on a GTX 1060, +7% on a Radeon 8060S. Where the default is already optimal (Radeon 890M) it changes
  nothing — which is the correct behaviour. `GGML_CUDA_TUNING=0` / `GGML_VK_TUNING=0` disable it.
- Fix: TQ2_1 aborted on the first batched matmul on Pascal, RDNA2 and RDNA3 GPUs.
- CUDA package 40% smaller; stripped binaries; NVIDIA EULA included in the CUDA packages.

## Quality
Same tasks, same seed, `llama-perplexity` 0-shot log-probability scoring (comparable **between rows**, not
with published leaderboard numbers):

| model | size | Winogrande | HellaSwag | MMLU (1548) |
|---|---|---|---|---|
| Qwen3.8-27B Q6_K_XL (parent) | 24.7 GB | 75.2 ± 1.9 | 82.3% | 42.0 ± 2.0 |
| Qwen3.8-27B Q2_K (conventional 2-bit) | 10.1 GB | 73.8 ± 2.0 | 76.3% | 38.7 ± 1.2 |
| **Bonsai-2-27B TQ2_1 / TQ1_1** | **6.7 / 5.5 GB** | 73.2 ± 2.0 | 75.6% | **41.7 ± 1.3** |

TQ1_1 and TQ2_1 give identical results (same weights). On knowledge the ternary model sits at the parent's
level and above conventional 2-bit (1.7σ), at 66% of its size. Winogrande cross-checked with the standard
lm-eval harness (0-shot, 1267 tasks): 73.5 ± 1.2. For the model's full benchmark suite (MMLU-Redux, GSM8K,
AIME, LiveCodeBench…) see the [Prism model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

## Known limitations (v0.2.0)
- **GTX 1060 / 6 GB GPUs with the 27B TQ1_1**: fits with ~44 of 64 layers on the GPU (`dartai-run.sh`
  computes it). Large-batch prefill can run out of VRAM in this version — use `-b 8 -ub 8`. Quality is
  identical; speed is CPU-bound.
- **TQ1_1 on CUDA** does prefill through the cuBLAS path (no MMQ yet): ~30% slower than TQ2_1 in batch. Decode
  is unaffected.
- Vulkan requires a **1.2+** driver (GTX 10xx: update the driver — the 2017 one exposes only Vulkan 1.0).

## Licenses and attribution
- llama.cpp / ggml: MIT (`LICENSE-llama.cpp`). The CUDA packages include NVIDIA runtime libraries under the
  NVIDIA EULA (`LICENSE-cuda-runtime.txt`, inside the package and in this repository).
- Bonsai models: Prism ML, Apache 2.0 — *Created using Bonsai by Prism ML.* Base Qwen3.8-27B: Alibaba Cloud,
  Apache 2.0.
- Format inspired by Georganas, Heinecke, Dubey, *Breaking the 1.58-bit Barrier for Ternary LLMs*
  (arXiv:2609.16338).

DartAI is the AI lab of the Dart group. Source code: closed at this stage.
