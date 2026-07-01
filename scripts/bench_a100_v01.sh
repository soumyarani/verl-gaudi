#!/bin/bash
#SBATCH -p public
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH -J bench_a100
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
SIF=/packages/apps/simg/pytorch_25.01-py3.sif
RAYTMP=/tmp/verl_ray_$SLURM_JOB_ID; mkdir -p $RAYTMP $WS/wandb
echo "NODE=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
apptainer exec --nv --cleanenv --bind /scratch:/scratch --bind /tmp:/tmp \
  --env PYTHONUSERBASE=$WS/cpkgs_cuda \
  --env HF_HOME=$WS/hf_cache --env HF_HUB_OFFLINE=1 \
  --env WANDB_API_KEY=$(cat $WS/.wandb_key) --env WANDB_DIR=$WS/wandb --env WANDB__SERVICE_WAIT=300 \
  --env TMPDIR=$RAYTMP --env RAY_TMPDIR=$RAYTMP \
  --env STEPS=${STEPS:-100} --env EXP=${EXP:-qwen0.5b_gsm8k_grpo_a100} \
  "$SIF" bash $WS/scripts/_bench_a100_inside_v01.sh
rm -rf $RAYTMP
echo "=== DONE bench_a100_v01 ==="
