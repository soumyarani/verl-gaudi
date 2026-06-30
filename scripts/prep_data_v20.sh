#!/bin/bash
#SBATCH -p gaudi
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -J verl_prep
#SBATCH --exclude=gaudi001
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
SIF=$WS/gaudi_124_pt210.sif
HLOG=$WS/run/habana_logs.$SLURM_JOB_ID; mkdir -p $HLOG $WS/data $WS/models $WS/hf_cache
echo "NODE=$(hostname) $(date)"
RUN="apptainer exec --cleanenv --no-home --bind /scratch:/scratch \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  --env HABANA_LOGS=$HLOG --env PYTHONUSERBASE=$WS/cpkgs \
  --env HF_HOME=$WS/hf_cache --env PT_HPU_LAZY_MODE=1 $SIF"
echo "############ GSM8k preprocess ############"
$RUN python3.10 $WS/verl/examples/data_preprocess/gsm8k.py --local_save_dir $WS/data/gsm8k 2>&1 | grep -viE "Lmod|Refreshing|modulefile|BRIDGE|^ PT_|Configuration|CPU Cores|CPU RAM|^---|^=|mark_step|lazy mode" | tail -8
ls -lh $WS/data/gsm8k/
echo "############ download Qwen2.5-0.5B-Instruct ############"
$RUN python3.10 -c "from huggingface_hub import snapshot_download; p=snapshot_download(\"Qwen/Qwen2.5-0.5B-Instruct\", local_dir=\"$WS/models/Qwen2.5-0.5B-Instruct\"); print(\"model at\", p)" 2>&1 | grep -viE "Lmod|Refreshing|modulefile" | tail -6
ls $WS/models/Qwen2.5-0.5B-Instruct/ | head
echo "=== DONE prep_data_v20 ==="
