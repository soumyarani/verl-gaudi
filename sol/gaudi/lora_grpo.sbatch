#!/bin/bash
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -J vg_lora
#SBATCH --exclude=gaudi005,gaudi008
#SBATCH --exclusive
# LoRA GRPO on 1 Gaudi2 (HL-225). Qwen2.5-0.5B GSM8k, LoRA-32 by default.
#
# Run:   sbatch sol/gaudi/lora_grpo.sbatch
# Tune:  LORA_RANK=64 LORA_ALPHA=64 STEPS=40 \
#          MODEL=$HOME/verl_gaudi/models/Qwen2.5-1.5B-Instruct sbatch sol/gaudi/lora_grpo.sbatch
set -uo pipefail
VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
source "$VG_HOME/sol/gaudi/env_common.sh"
vg_gaudi_run "$VG_HOME/sol/gaudi/_inside_lora_grpo.sh"
echo "=== DONE vg_lora ==="
