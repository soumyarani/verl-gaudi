#!/bin/bash
# Build the native CUDA verl venv for the A100 path (idempotent).
# The venv (~8G) lives on scratch under $VG_WORK because it is fully rebuildable from here.
# Run once on an A100 node (or let the sbatch wrappers call it automatically).
set -uo pipefail
VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
VG_WORK="${VG_WORK:-/scratch/$USER/verl_gaudi_work}"
UVBIN="$VG_WORK/uvbin"
VENV="$VG_WORK/venv_a100native"
mkdir -p "$VG_WORK"

if [ ! -x "$UVBIN/uv" ]; then
  echo "installing uv into $UVBIN ..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UVBIN" sh 2>&1 | tail -2
fi
export PATH="$UVBIN:$PATH"

if [ ! -d "$VENV" ]; then
  echo "creating venv + installing verl + vLLM (slow first run) ..."
  uv venv "$VENV" --python 3.12 2>&1 | tail -2
  source "$VENV/bin/activate"
  uv pip install "vllm==0.8.5" 2>&1 | tail -3
  uv pip install -e "$VG_HOME/runtime/verl_cuda" --no-deps 2>&1 | tail -2
  uv pip install "tensordict==0.8.3" "ray[default]" hydra-core omegaconf codetiming dill \
    "pyarrow>=19" pandas pylatexenc "datasets==5.0.0" peft torchdata wandb math-verify \
    cachetools fastapi uvicorn pydantic cloudpickle orjson accelerate 2>&1 | tail -3
else
  source "$VENV/bin/activate"
fi
python -c "import torch,vllm,verl,transformers; print('torch',torch.__version__,'vllm',vllm.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
echo "A100 venv ready: $VENV"
