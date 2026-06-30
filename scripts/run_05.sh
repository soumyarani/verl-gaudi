#!/bin/bash
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0:55:00
#SBATCH -J verl_grpo
#SBATCH --exclude=gaudi001,gaudi002,gaudi003,gaudi006,gaudi008,gaudi009
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
SIF=$WS/gaudi_124_pt210.sif
HLOG=$WS/run/habana_logs.$SLURM_JOB_ID; mkdir -p $HLOG
RAYTMP=/tmp/verl_ray_$SLURM_JOB_ID; mkdir -p $RAYTMP
echo "NODE=$(hostname) $(date)"
apptainer exec --cleanenv --no-home --bind /scratch:/scratch --bind /tmp:/tmp \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  --env HABANA_LOGS=$HLOG --env PYTHONUSERBASE=$WS/cpkgs \
  --env HF_HOME=$WS/hf_cache --env HF_HUB_OFFLINE=1 \
  --env PT_HPU_LAZY_MODE=1 --env PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  --env VERL_PLATFORM=hpu --env TMPDIR=$RAYTMP --env RAY_TMPDIR=$RAYTMP \
  --env RAY_agent_register_timeout_ms=300000 \
  --env RAY_raylet_start_wait_time_s=600 \
  --env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  --env PATH=$WS/cpkgs/bin:/usr/local/bin:/usr/bin:/bin \
  "$SIF" bash $WS/scripts/_run_inside_05.sh
echo "=== capturing ray session logs ==="
SL=$WS/logs/raylogs_$SLURM_JOB_ID; mkdir -p $SL
cp -f $RAYTMP/session_latest/logs/raylet.* $SL/ 2>/dev/null || true
cp -f $RAYTMP/session_latest/logs/gcs_server.* $SL/ 2>/dev/null || true
cp -f $RAYTMP/session_latest/logs/dashboard* $SL/ 2>/dev/null || true
ls -la $SL 2>/dev/null
rm -rf $RAYTMP
echo "=== DONE run_grpo_v41 ==="
