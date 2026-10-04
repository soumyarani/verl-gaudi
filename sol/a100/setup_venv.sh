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
  # Pin transformers LAST: vllm 0.8.5 otherwise resolves transformers 5.x, which drops
  # AutoModelForVision2Seq (verl 0.5 imports it) and is incompatible with vllm 0.8.5.
  uv pip install "transformers==4.51.3" 2>&1 | tail -2
else
  source "$VENV/bin/activate"
fi

# verl 0.5's dp_actor.py imports flash-attn unconditionally on CUDA; we run with
# use_remove_padding=False (its symbols are then unused), so make the import optional
# rather than building flash-attn. Idempotent.
DPA="$VG_HOME/runtime/verl_cuda/verl/workers/actor/dp_actor.py"
if [ -f "$DPA" ] && ! grep -q "except ModuleNotFoundError" "$DPA"; then
  python - "$DPA" <<'PY'
import sys
f=sys.argv[1]; s=open(f).read()
old="    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\nelif is_npu_available:"
new=("    try:\n"
     "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
     "    except ModuleNotFoundError:\n"
     "        index_first_axis = pad_input = rearrange = unpad_input = None  # unused when use_remove_padding=False\n"
     "elif is_npu_available:")
if old in s:
    open(f,"w").write(s.replace(old,new,1)); print("patched dp_actor flash-attn import")
else:
    print("dp_actor pattern not found (already patched or verl changed)")
PY
fi

# verl 0.5 HARD-CODES attn_implementation="flash_attention_2" in fsdp_workers.py. Without a
# flash-attn build, transformers errors ("FlashAttention2 toggled on but not installed").
# Switch to sdpa (correct; slower on long sequences — install flash-attn + revert to FA2 for perf).
FSW="$VG_HOME/runtime/verl_cuda/verl/workers/fsdp_workers.py"
if [ -f "$FSW" ]; then
  sed -i 's|attn_implementation="flash_attention_2"|attn_implementation="sdpa"|g' "$FSW"
fi
python -c "import torch,vllm,verl,transformers; print('torch',torch.__version__,'vllm',vllm.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
echo "A100 venv ready: $VENV"
