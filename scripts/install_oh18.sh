#!/bin/bash
#SBATCH -p htc
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -J oh18
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi; SIF=$WS/gaudi_124_pt210.sif; CP=$WS/cpkgs
RUN="apptainer exec --cleanenv --no-home --bind /scratch:/scratch --env PYTHONUSERBASE=$CP $SIF"
echo "=== install matched transformers 4.49 + optimum-habana 1.18 (no torch change) ==="
$RUN python3.10 -m pip install --user -c $WS/cconstraints.txt "transformers==4.49.0" "tokenizers" 2>&1 | grep -viE "already satisfied" | tail -8
$RUN python3.10 -m pip install --user --no-deps "optimum-habana==1.18.0" 2>&1 | tail -3
echo "=== test adapt import ==="
$RUN bash -lc "export VERL_PLATFORM=hpu; PT_HPU_LAZY_MODE=1 python3.10 -c \"
import habana_frameworks.torch
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi(); print(\\\"ADAPT OK\\\")
import transformers; print(\\\"transformers\\\", transformers.__version__)
\"" 2>&1 | grep -viE "mark_step|lazy mode|Lmod|Refreshing|warn|FutureWarn" | tail -8
echo "=== DONE oh18 ==="
