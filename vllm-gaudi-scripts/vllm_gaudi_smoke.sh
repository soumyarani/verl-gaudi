#!/bin/bash
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -J vllm_gaudi_smoke
#SBATCH --exclusive
#SBATCH --exclude=gaudi005,gaudi008
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
SIF=$WS/gaudi_124_vllm.sif
HLOG=$WS/run/habana_logs.$SLURM_JOB_ID; mkdir -p $HLOG
VCACHE=$WS/vllm_cache; mkdir -p $VCACHE
echo "NODE=$(hostname) $(date)"
apptainer exec --cleanenv --no-home --bind /scratch:/scratch --bind /tmp:/tmp \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  --env HABANA_LOGS=$HLOG --env PYTHONUSERBASE=$WS/cpkgs_vllm \
  --env HF_HOME=$WS/hf_cache --env HF_HUB_OFFLINE=1 \
  --env XDG_CACHE_HOME=$VCACHE --env VLLM_CACHE_ROOT=$VCACHE --env HOME=$VCACHE \
  --env PT_HPU_LAZY_MODE=1 --env PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  --env VLLM_SKIP_WARMUP=true --env VLLM_ENABLE_V1_MULTIPROCESSING=0 --env VLLM_WORKER_MULTIPROC_METHOD=fork \
  --env PATH=$WS/cpkgs_vllm/bin:/usr/local/bin:/usr/bin:/bin \
  "$SIF" /usr/bin/python3.10 $WS/scripts/vllm_gaudi_smoke.py
echo "=== DONE vllm_gaudi_smoke ==="
