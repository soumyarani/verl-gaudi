#!/bin/bash
# FULL GRPO on GSM8k, EAGER mode (all 3 HPU blockers solved). Disaggregated, 4 HPU.
# 200 steps, flexible reward, wandb, checkpoint every 20 + resume_mode=auto, retry wrapper for safety.
export VERL_PLATFORM=hpu PT_HPU_LAZY_MODE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=0 VLLM_SKIP_WARMUP=true HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline WANDB_DIR=/workspace/wandb WANDB_PROJECT=verl_air_gaudi
source /workspace/venv/bin/activate
mkdir -p /workspace/wandb /workspace/ckpt
CKPT=/workspace/ckpt/grpo_eager_full_v08
run_once () {
  cd /root
  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/data/gsm8k/train.parquet data.val_files=/workspace/data/gsm8k/test.parquet \
    data.train_batch_size=16 data.max_prompt_length=256 data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=bfloat16 actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend=hccl \
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.plugin.checkpoint_engine.hccl_hpu \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=768 \
    actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    ray_kwargs.ray_init.num_cpus=64 +ray_kwargs.ray_init.num_gpus=0 \
    +ray_kwargs.ray_init.resources="{HPU:4}" +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
    trainer.logger=[console,wandb] trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=hpu \
    trainer.use_v1=True trainer.v1.trainer_mode=separate_async \
    trainer.default_local_dir=$CKPT trainer.resume_mode=auto \
    trainer.save_freq=20 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps=200 \
    trainer.project_name=verl_air_gaudi trainer.experiment_name=grpo_eager_full_v08
}
cleanup () { ray stop --force >/dev/null 2>&1; for p in $(hl-smi 2>/dev/null | sed -n '/Compute Processes/,/^+====/p' | awk '{print $3}' | grep -E '^[0-9]+$'); do kill -9 "$p" 2>/dev/null; done; sleep 8; }
for attempt in $(seq 1 8); do
  echo "======== ATTEMPT $attempt ($(date +%H:%M:%S)) ========"
  run_once; RC=$?
  echo "ATTEMPT_${attempt}_RC=$RC"
  [ $RC -eq 0 ] && { echo "ALL_DONE_RC0"; break; }
  echo "crash RC=$RC; cleanup + resume"; cleanup
done
echo "EAGER_FULL_FINISHED"
