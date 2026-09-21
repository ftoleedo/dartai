# DartAI Runtime — Ternary LLMs on llama.cpp, any GPU

Rode um **Qwen3.8-27B ternário em 5,5–6,7 GB** — em NVIDIA (CUDA), AMD/Intel/NVIDIA (Vulkan) ou CPU.
Binários prontos; nenhuma compilação. Os formatos **TQ1_1** (1,75 bpw) e **TQ2_1** (2,13 bpw) são
bit-exatos ao checkpoint ternário do Prism.

## Pacotes (v0.2.0)
| arquivo | para | tamanho |
|---|---|---|
| `dartai-linux-x64-vulkan-0.2.0.tar.gz` | Linux x86-64 (glibc ≥ 2.35), qualquer GPU com driver Vulkan 1.2+ (AMD, Intel, NVIDIA incl. GTX 10xx) e CPU | 32 MB |
| `dartai-linux-x64-cuda12-0.2.0.tar.gz` | Linux x86-64, NVIDIA GTX 10xx → RTX 50xx (sm 61–120), runtime CUDA 12 incluído · driver ≥ 525 | 849 MB |
| `dartai-windows-x64-vulkan-0.2.0.zip` | Windows 10/11 x64, qualquer GPU com driver Vulkan 1.2+ | 26 MB |
| `dartai-windows-x64-cuda12-0.2.0.zip` | Windows 10/11 x64, NVIDIA GTX 10xx → RTX 50xx, DLLs CUDA 12 incluídas · driver ≥ 525 | 555 MB |

Confira os hashes em `SHA256SUMS`. Os pacotes Windows foram compilados com MSVC e validados numa GTX 1060
(corretude dos kernels ternários e sintonizador).

## Modelos (Hugging Face, Apache 2.0 — não redistribuídos aqui)
| modelo | formato | tamanho | quando |
|---|---|---|---|
| [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | **TQ2_1** | 6,7 GB | GPU ≥ 8 GB, iGPU/RAM unificada ≥ 12 GB — **o mais rápido** |
| Ternary-Bonsai-2-27B | **TQ1_1** | 5,5 GB | o menor; GPU de 6 GB com descarga parcial (ver limitações) |
| Ternary-Bonsai-1.7B | TQ2_1 | 0,46 GB | teste rápido / máquinas fracas |

## Uso (Linux)
```bash
tar xzf dartai-linux-x64-cuda12-0.2.0.tar.gz && cd dartai-linux-x64-cuda12
./dartai-run.sh ~/models/Ternary-Bonsai-2-27B-TQ2_1.gguf 8080 8192   # escolhe -ngl pela VRAM livre
# API OpenAI-compatível em http://127.0.0.1:8080/v1 ; UI em http://127.0.0.1:8080
```
Binário direto: `LD_LIBRARY_PATH=. ./llama-server -m modelo.gguf -ngl 99 -c 8192 --port 8080`.

**Windows**: descompacte o zip e rode `llama-server.exe -m modelo.gguf -ngl 99 -c 8192 --port 8080` de dentro da pasta (as DLLs ficam ao lado).

## O que há de novo na 0.2.0
- **Sintonizador de kernel por medição** (CUDA e Vulkan): na primeira inferência o runtime mede as
  alternativas de kernel *na sua GPU* e guarda a escolha em `~/.cache/dartai/`. Medido: +8–12% de decode
  e **+117% no especulativo** numa RTX 5070 Ti; **+22%** numa GTX 1060; +7% numa Radeon 8060S. Onde o
  padrão já é o ótimo (Radeon 890M) ele não muda nada. `GGML_CUDA_TUNING=0` / `GGML_VK_TUNING=0` desligam.
- Correção: TQ2_1 abortava no primeiro matmul em lote em GPUs Pascal, RDNA2 e RDNA3.
- Pacote CUDA 40% menor; binários stripped.

## Medido (20/09/2026, Bonsai-2-27B, decode `tg128`, na tomada, perfil performance)
| hardware | backend | TQ2_1 | TQ1_1 |
|---|---|---|---|
| RTX 5070 Ti Laptop 12 GB | CUDA | **44,4 t/s** | 27,7 t/s |
| AMD Ryzen AI MAX+ 395 (Radeon 8060S, memória unificada) | Vulkan | 23,2 t/s | 19,7 t/s |
| GTX 1060 6 GB | CUDA | não cabe | roda com 44/64 camadas na GPU (ver limitações) |

Especulativo (draft DFlash2 Q4_K_M, `--spec-type draft-dflash`): **111,7 t/s** em código na 5070 Ti.
Em iGPU o especulativo **piora** (−13%) — não use lá.

## Qualidade
Mesmas tarefas, mesma semente, `llama-perplexity` 0-shot por logprob (comparável **entre linhas**, não
com números publicados):

| modelo | tamanho | Winogrande | HellaSwag | MMLU (1548) |
|---|---|---|---|---|
| Qwen3.8-27B Q6_K_XL (pai) | 24,7 GB | 75,2 ± 1,9 | 82,3% | 42,0 ± 2,0 |
| Qwen3.8-27B Q2_K (2-bit convencional) | 10,1 GB | 73,8 ± 2,0 | 76,3% | 38,7 ± 1,2 |
| **Bonsai-2-27B TQ2_1 / TQ1_1** | **6,7 / 5,5 GB** | 73,2 ± 2,0 | 75,6% | **41,7 ± 1,3** |

TQ1_1 e TQ2_1 dão o mesmo resultado (mesmos pesos). Em conhecimento o ternário fica no nível do pai e
acima do 2-bit convencional (1,7σ), com 66% do tamanho deste. Para os benchmarks completos do modelo
(MMLU-Redux, GSM8K, AIME, LiveCodeBench…) veja o
[model card do Prism](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

## Limitações conhecidas (v0.2.0)
- **GTX 1060 / placas de 6 GB com o 27B TQ1_1**: cabe com ~44 das 64 camadas na GPU (`dartai-run.sh`
  calcula). Em prefill de lote grande pode estourar a VRAM nesta versão — use `-b 8 -ub 8`. Qualidade
  idêntica; velocidade limitada pela CPU.
- **TQ1_1 no CUDA** faz prefill pelo caminho cuBLAS (sem MMQ ainda): ~30% mais lento que o TQ2_1 em
  lote. Decode não é afetado.
- Vulkan exige driver **1.2+** (GTX 10xx: atualize o driver — o de 2017 expõe só Vulkan 1.0).

## Licenças e atribuições
- llama.cpp / ggml: MIT (`LICENSE-llama.cpp`). O pacote CUDA inclui bibliotecas de runtime NVIDIA sob a
  EULA da NVIDIA (`LICENSE-cuda-runtime.txt` dentro do pacote).
- Modelos Bonsai: Prism ML, Apache 2.0 — *Created using Bonsai by Prism ML.* Base Qwen3.8-27B: Alibaba
  Cloud, Apache 2.0.
- Formato inspirado em Georganas, Heinecke, Dubey, *Breaking the 1.58-bit Barrier for Ternary LLMs*
  (arXiv:2609.16338).

DartAI é o laboratório de IA do grupo Dart. Código-fonte: fechado nesta fase.
