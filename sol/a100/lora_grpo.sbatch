#!/bin/bash
#SBATCH -p public
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=110G
#SBATCH -t 2:00:00
#SBATCH -J a100_lora
# LoRA GRPO on 1 A100 (native vLLM rollout). Qwen2.5-0.5B GSM8k, LoRA-32 by default.
#
# Run:   sbatch sol/a100/lora_grpo.sbatch
# Tune:  LORA_RANK=64 LORA_ALPHA=64 STEPS=40 sbatch sol/a100/lora_grpo.sbatch
set -uo pipefail
VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
VG_WORK="${VG_WORK:-/scratch/$USER/verl_gaudi_work}"
export VG_HOME VG_WORK
export HF_HOME="$VG_WORK/hf_cache" HF_HUB_OFFLINE=1
[ -f "$VG_HOME/.wandb_key" ] && export WANDB_API_KEY="$(cat "$VG_HOME/.wandb_key")"
export WANDB_DIR="$VG_WORK" WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/verl_ray_$SLURM_JOB_ID"; mkdir -p "$TMPDIR" "$VG_WORK/hf_cache"
export RAY_TMPDIR="$TMPDIR"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
echo "NODE=$(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash "$VG_HOME/sol/a100/_inside_lora_grpo.sh"
rm -rf "$TMPDIR"
echo "=== DONE a100_lora ==="
