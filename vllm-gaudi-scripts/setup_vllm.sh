#!/bin/bash
#SBATCH -p htc
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0:45:00
#SBATCH -J setup_vllm
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
SIF=$WS/gaudi_124_vllm.sif
CP=$WS/cpkgs_vllm; mkdir -p $CP
echo "NODE=$(hostname) $(date)"
RUN="apptainer exec --cleanenv --no-home --bind /scratch:/scratch --env PYTHONUSERBASE=$CP $SIF"
echo "=== build constraints from container stack ==="
$RUN python3.10 - <<PY
import importlib.metadata as m
keep=["torch","vllm","vllm_gaudi","transformers","tensordict","numpy","pandas","torchvision","torchaudio"]
out=[]
for k in keep:
    try: out.append(f"{k}=={m.version(k)}")
    except Exception: pass
open("$WS/cv_constraints.txt","w").write("\n".join(out)+"\n")
print("\n".join(out))
PY
echo "=== install verl (no-deps) + ray[default] 2.47.1 + pure-python deps ==="
$RUN python3.10 -m pip install --user --no-deps -e $WS/verl 2>&1 | tail -3
$RUN python3.10 -m pip install --user -c $WS/cv_constraints.txt "ray[default]==2.47.1" 2>&1 | grep -viE "already satisfied" | tail -6
$RUN python3.10 -m pip install --user -c $WS/cv_constraints.txt \
  accelerate codetiming datasets dill hydra-core omegaconf peft pyarrow pybind11 pylatexenc \
  torchdata wandb msgspec packaging tqdm cachetools uvicorn fastapi latex2sympy2_extended \
  math_verify tensorboard pyvers cloudpickle importlib_metadata orjson 2>&1 | grep -viE "already satisfied" | tail -20
echo "=== verify import verl + vllm + platform ==="
$RUN bash -lc "export VERL_PLATFORM=hpu; python3.10 -c \"import habana_frameworks.torch; import vllm; print('vllm', vllm.__version__); import verl; from verl.plugin.platform import get_platform; print('platform', get_platform().device_name); import transfer_queue\" 2>&1 | grep -viE 'mark_step|lazy mode|Refreshing|Lmod' | tail -8"
echo "=== DONE setup_vllm ==="
