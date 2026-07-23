#!/bin/bash
# Qwen3-8B LoRA GRPO on the FULL LiveCodeBench, hyperparameters matched to SDPO (lasgroup/SDPO
# experiments/rich_feedback baseline_grpo + user.yaml), adapted for our Gaudi HPU disaggregated stack.
# HPU-forced deviations vs SDPO (which is full-param on 4 GPUs): LoRA(merge) + bf16 + eager + bucketed sync.
# NOTE: max_response=8192 is very aggressive on HPU (device-frag/memory) — reduce if it won't fit (iterate).
export VERL_PLATFORM=hpu PT_HPU_LAZY_MODE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=0 VLLM_SKIP_WARMUP=true HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline WANDB_DIR=/workspace/wandb WANDB_PROJECT=verl_air_lcb
source /workspace/venv/bin/activate
cd /root
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  data.train_files=/workspace/data/lcb_full/train.parquet data.val_files=/workspace/data/lcb_full/test.parquet \
  data.train_batch_size=32 data.max_prompt_length=2048 data.max_response_length=8192 \
  data.filter_overlong_prompts=True data.truncation=right \
  data.apply_chat_template_kwargs='{enable_thinking:false}' \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.lora_rank=32 actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.dtype=bfloat16 actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.temperature=0.6 actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.max_model_len=10240 actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=hccl \
  actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.plugin.checkpoint_engine.hccl_hpu \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1536 \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  custom_reward_function.path=/workspace/lcb_reward.py custom_reward_function.name=compute_score \
  ray_kwargs.ray_init.num_cpus=16 +ray_kwargs.ray_init.num_gpus=0 \
  +ray_kwargs.ray_init.resources="{HPU:4}" +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  trainer.logger=[console,wandb] trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=hpu \
  trainer.use_v1=True trainer.v1.trainer_mode=separate_async \
  trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=30 \
  trainer.project_name=verl_air_lcb trainer.experiment_name=lcb_8b_lora_full
echo "VERL_RC=$?"
