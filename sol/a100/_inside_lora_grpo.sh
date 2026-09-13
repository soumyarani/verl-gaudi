#!/bin/bash
# A100 LoRA GRPO, Qwen2.5 on GSM8k, NATIVE vLLM rollout. vLLM serves the LoRA adapter
# directly on CUDA (rollout.load_format=safetensors + layered_summon), so no HF-rollout
# fallback is needed here — unlike the Gaudi path.
set -uo pipefail
VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
VG_WORK="${VG_WORK:-/scratch/$USER/verl_gaudi_work}"
MODEL="${MODEL:-$VG_HOME/models/Qwen2.5-0.5B-Instruct}"
STEPS="${STEPS:-5}"
TBS="${TBS:-64}"; MINI="${MINI:-32}"; RESP="${RESP:-512}"
LORA_RANK="${LORA_RANK:-32}"; LORA_ALPHA="${LORA_ALPHA:-32}"
EXP="${EXP:-a100_lora${LORA_RANK}_grpo_$(basename "$MODEL")}"

source "$VG_HOME/sol/a100/setup_venv.sh"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$VG_HOME/data/gsm8k/train.parquet" \
  data.val_files="$VG_HOME/data/gsm8k/test.parquet" \
  data.train_batch_size="$TBS" data.max_prompt_length=512 data.max_response_length="$RESP" \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  algorithm.use_kl_in_reward=False ray_init.num_cpus=8 \
  trainer.logger='["console","wandb"]' trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=cuda \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps="$STEPS" \
  trainer.project_name=verl_gaudi trainer.experiment_name="$EXP"
echo "VERL_RC=$?"
