#!/bin/bash
#SBATCH -p htc
#SBATCH -q public
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -o /scratch/ssamine4/verl_gaudi/logs/dl15b-%j.out
set -eux
source /scratch/ssamine4/hinge-loss/env/bin/activate
python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/scratch/ssamine4/verl_gaudi/models/Qwen2.5-1.5B-Instruct", ignore_patterns=["*.pth","original/*"])
print("MODEL15B_DONE")
PY
