#!/bin/bash
#SBATCH -p public
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=110G
#SBATCH -t 3:00:00
#SBATCH -J a100_vllm
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
VENV=$WS/venv_a100native
export PATH=$WS/uvbin:$PATH
export HF_HOME=$WS/hf_cache HF_HUB_OFFLINE=1
export WANDB_API_KEY=$(cat $WS/.wandb_key) WANDB_DIR=$WS/wandb WANDB__SERVICE_WAIT=300
export TMPDIR=/tmp/verl_ray_$SLURM_JOB_ID; mkdir -p $TMPDIR $WS/wandb
export RAY_TMPDIR=$TMPDIR
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
echo "NODE=$(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- one-time env build with uv (idempotent) ---
if [ ! -x "$WS/uvbin/uv" ]; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$WS/uvbin" sh 2>&1 | tail -2
fi
if [ ! -d "$VENV" ]; then
  echo "creating venv + installing verl + vllm (this is the slow first run)..."
  uv venv "$VENV" --python 3.12 2>&1 | tail -2
  source "$VENV/bin/activate"
  uv pip install "vllm==0.8.5" 2>&1 | tail -3
  uv pip install -e "$WS/verl_a100_native" --no-deps 2>&1 | tail -2
  uv pip install "tensordict==0.8.3" "ray[default]" hydra-core omegaconf codetiming dill \
    "pyarrow>=19" pandas pylatexenc "datasets==5.0.0" peft torchdata wandb math-verify \
    cachetools fastapi uvicorn pydantic cloudpickle orjson accelerate 2>&1 | tail -3
else
  source "$VENV/bin/activate"
fi
python -c "import torch,vllm,verl,transformers; print('torch',torch.__version__,'vllm',vllm.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"

STEPS=${STEPS:-2}
# --- native verl + vLLM rollout, FULL verl GSM8k GRPO settings ---
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$WS/data/gsm8k/train.parquet data.val_files=$WS/data/gsm8k/test.parquet \
  data.train_batch_size=1024 data.max_prompt_length=512 data.max_response_length=1024 \
  actor_rollout_ref.model.path=$WS/models/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  algorithm.use_kl_in_reward=False ray_init.num_cpus=8 \
  trainer.logger=[console,wandb] trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=cuda \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=$STEPS \
  trainer.project_name=verl-gaudi-vs-a100 trainer.experiment_name=qwen0.5b_gsm8k_grpo_a100_vllm_FULL
echo "VERL_RC=$?"
rm -rf $TMPDIR
echo "=== DONE a100_vllm_v01 ==="
