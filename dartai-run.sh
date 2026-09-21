#!/bin/bash
# dartai-run.sh — sobe o llama-server do DartAI Runtime escolhendo backend e -ngl pela GPU detectada.
# uso: dartai-run.sh <modelo.gguf> [porta] [ctx]        (rode de dentro da pasta do pacote, ou exporte DARTAI_HOME)
set -u
M=${1:?modelo.gguf}; PORT=${2:-8080}; CTX=${3:-8192}
HOME_=${DARTAI_HOME:-$(cd "$(dirname "$0")" && pwd)}
SIZE=$(stat -c %s "$M" 2>/dev/null || echo 0); SIZE_GB=$(python3 -c "print(round($SIZE/1024**3,2))" 2>/dev/null || echo "?")
echo "== DartAI Runtime · $(basename "$M") ($SIZE_GB GiB) · ctx $CTX"
if command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.free --format=csv,noheader >/dev/null 2>&1; then
  IFS=, read -r GPU FREE <<<"$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1)"
  FREE_MB=${FREE//[^0-9]/}; echo "GPU: $GPU · VRAM livre ${FREE_MB} MiB"
  # layers cabem? pesos + KV(~ctx*0.06 MB) + 300 MB compute; 64 camadas no 27B
  NEED_MB=$(( SIZE/1048576 + CTX*64/1000 + 300 ))
  if [ "$NEED_MB" -le "$FREE_MB" ]; then NGL=99; else NGL=$(( 64 * FREE_MB / NEED_MB - 2 )); [ "$NGL" -lt 0 ] && NGL=0; fi
  echo "-ngl $NGL (precisa ~${NEED_MB} MiB para tudo na GPU)"
  BIN=$HOME_/llama-server; export LD_LIBRARY_PATH=$HOME_:${LD_LIBRARY_PATH:-}
else
  echo "GPU: sem NVIDIA → Vulkan (AMD/Intel/iGPU) ou CPU"; NGL=99; BIN=$HOME_/llama-server; export LD_LIBRARY_PATH=$HOME_:${LD_LIBRARY_PATH:-}
fi
exec "$BIN" -m "$M" -c "$CTX" -ngl "$NGL" --port "$PORT" --host 127.0.0.1 -fa on "${@:4}"
