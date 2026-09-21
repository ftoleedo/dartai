# DartAI Runtime — Ternary LLMs on llama.cpp, any GPU

> **English summary.** DartAI Runtime is a build of [llama.cpp](https://github.com/ggml-org/llama.cpp) with two
> new **ternary** quantization formats — **TQ1_1** (1.75 bits/weight) and **TQ2_1** (2.13 bits/weight) — plus
> CPU, Vulkan and CUDA kernels for them and a **runtime kernel autotuner** that measures your GPU on first use.
> It runs a **27B-parameter model (Qwen3.8-27B ternary, "Bonsai 2" by Prism ML) in 5.5–6.7 GB**, on NVIDIA
> (GTX 10xx → RTX 50xx), AMD (RDNA, Strix Halo), Intel or CPU — Linux and Windows, no compilation, no Python.
> OpenAI-compatible API out of the box. Ternary quality measured against the FP16 parent: within 0–2 points on
> Winogrande and MMLU, on par with conventional 2-bit (Q2_K) at 66% of its size.
> *Keywords: ternary LLM · 1.58-bit · 2-bit quantization · BitNet-style · low VRAM · 6 GB GPU · GTX 1060 ·
> llama.cpp · GGUF · Vulkan · CUDA · AMD Strix Halo · Radeon 8060S · Qwen3.8-27B · Bonsai · Prism ML ·
> local LLM · offline inference · OpenAI-compatible server · speculative decoding · kernel autotuning.*

## O que é isto

**DartAI Runtime** é o [llama.cpp](https://github.com/ggml-org/llama.cpp) — o motor de inferência de LLMs
locais mais usado do mundo — com dois formatos de peso novos, **TQ1_1** e **TQ2_1**, e os kernels que os
fazem rodar rápido em GPU. Você baixa um pacote, baixa um modelo `.gguf` ternário e roda um servidor com
API compatível com a da OpenAI na sua máquina. Sem instalar Python, CUDA ou driver especial; sem nuvem.

**Para quem é**: quem quer rodar um modelo de **27 bilhões de parâmetros** numa placa comum — uma GTX 1060
de 6 GB, um notebook com Radeon integrada, uma RTX de 8–12 GB — e não cabe com os formatos convencionais.
O mesmo Qwen3.8-27B em Q4_K_M precisa de 15,6 GB; em Q2_K, 10,1 GB; em **TQ1_1, 5,5 GB**.

## O que é "ternário"

Um LLM comum guarda cada peso como um número de 16 bits. Quantização é reduzir isso: 8, 4, 2 bits por
peso, trocando precisão por memória. **Ternário** é o extremo prático: cada peso só pode valer
**−1, 0 ou +1** — três estados, o que dá log₂(3) ≈ 1,58 bits de informação por peso. Com o empacotamento
e as escalas por bloco, os formatos DartAI ficam em **1,75 bpw (TQ1_1)** e **2,13 bpw (TQ2_1)**.

Duas coisas fazem isso funcionar na prática, e as duas estão medidas neste repositório:

- **O modelo tem que ser treinado ternário**, não convertido depois. Um modelo pós-quantizado a 2 bits
  (IQ2_XXS, Q2_K) degrada; o Bonsai 2 da Prism ML é *treinado* com pesos ternários e segura a qualidade
  do pai — é o checkpoint que este runtime roda. Por isso o TQ1_1 e o TQ2_1 dão exatamente a mesma
  qualidade: são duas codificações *lossless* dos mesmos três estados.
- **O kernel tem que aproveitar os bytes a menos.** Decode de LLM é limitado por banda de memória; ler
  1,75 bits em vez de 16 por peso é o que faz um 27B decodificar a 44 t/s numa RTX 5070 Ti de notebook. Em
  GPUs limitadas por latência (iGPUs), o trabalho de desempacotar os trits pesa mais — por isso o
  sintonizador mede a sua GPU em vez de assumir.

O que este runtime **não** é: não é um quantizador (você não converte seus próprios modelos para ternário
aqui — isso é treino), e não é um app com interface — para isso existe o LlamaForge by DartAI.

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
