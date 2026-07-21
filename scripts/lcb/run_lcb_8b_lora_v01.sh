#!/bin/bash
# Qwen3-8B LoRA GRPO on LiveCodeBench (SDPO data+reward) on Gaudi (eager, disaggregated, 4 HPU).
# Conservative seq lengths to de-risk HPU memory/fragmentation first. 3 steps.
export VERL_PLATFORM=hpu PT_HPU_LAZY_MODE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=0 VLLM_SKIP_WARMUP=true HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline WANDB_DIR=/workspace/wandb WANDB_PROJECT=verl_air_lcb
source /workspace/venv/bin/activate
cd /root
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/workspace/data/lcb_v6/train.parquet data.val_files=/workspace/data/lcb_v6/test.parquet \
  data.train_batch_size=8 data.max_prompt_length=1024 data.max_response_length=1024 \
  data.filter_overlong_prompts=True data.truncation=right \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.dtype=bfloat16 actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.checkpoint_engine.backend=hccl \
  actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.plugin.checkpoint_engine.hccl_hpu \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1536 \
  actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_model_len=2560 \
  custom_reward_function.path=/workspace/lcb_reward.py custom_reward_function.name=compute_score \
  algorithm.use_kl_in_reward=False \
  ray_kwargs.ray_init.num_cpus=64 +ray_kwargs.ray_init.num_gpus=0 \
  +ray_kwargs.ray_init.resources="{HPU:4}" +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  trainer.logger=[console,wandb] trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=hpu \
  trainer.use_v1=True trainer.v1.trainer_mode=separate_async \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps=3 \
  trainer.project_name=verl_air_lcb trainer.experiment_name=lcb_8b_lora_v01
echo "VERL_RC=$?"
