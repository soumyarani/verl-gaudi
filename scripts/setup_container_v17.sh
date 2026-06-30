#!/bin/bash
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0:45:00
#SBATCH -J verl_csetup
#SBATCH --exclude=gaudi001
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
SIF=$WS/gaudi_124_pt210.sif
HLOG=$WS/run/habana_logs.$SLURM_JOB_ID; mkdir -p $HLOG
echo "NODE=$(hostname) $(date)"
RUN="apptainer exec --cleanenv --no-home --bind /scratch:/scratch \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  --env HABANA_LOGS=$HLOG \
  --env PYTHONUSERBASE=$WS/cpkgs \
  --env PT_HPU_LAZY_MODE=1 $SIF"
echo "############ INSTALL ############"
$RUN bash $WS/scripts/_install_inside.sh 2>&1 | grep -viE "Lmod|Refreshing|modulefile|already satisfied" | tail -35
echo "############ VERIFY ############"
$RUN python3.10 $WS/scripts/_verify_inside.py 2>&1 \
  | grep -viE "mark_step|add_step_closure|lazy mode only|^ PT_|Configuration|CPU Cores|CPU RAM|^---|^=|BRIDGE|Lmod|Refreshing|modulefile" | tail -20
echo "=== DONE setup_container_v17 ==="
