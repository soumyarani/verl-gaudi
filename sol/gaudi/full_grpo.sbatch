#!/bin/bash
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0:55:00
#SBATCH -J vg_full
#SBATCH --exclude=gaudi005,gaudi008
#SBATCH --exclusive
# Full-parameter GRPO on 1 Gaudi2 (HL-225). Qwen2.5-0.5B GSM8k by default.
#
# Run:   sbatch sol/gaudi/full_grpo.sbatch
# Tune:  STEPS=10 MODEL=$HOME/verl_gaudi/models/Qwen2.5-1.5B-Instruct sbatch sol/gaudi/full_grpo.sbatch
#        (--export=ALL is the sbatch default, so exported env vars reach the job)
set -uo pipefail
VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
source "$VG_HOME/sol/gaudi/env_common.sh"
vg_gaudi_run "$VG_HOME/sol/gaudi/_inside_full_grpo.sh"
echo "=== DONE vg_full ==="
