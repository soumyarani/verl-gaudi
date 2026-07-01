#!/bin/bash
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
STEPS=${STEPS:-100}
EXP=${EXP:-qwen0.5b_gsm8k_grpo_a100}
export PYTHONUSERBASE=$WS/cpkgs_cuda
echo "=== A100 run: STEPS=$STEPS EXP=$EXP cpkgs=$PYTHONUSERBASE ==="
# install verl + deps into the user-site (idempotent; torch/numpy pinned to container)
python3 -m pip install --user --no-deps -e $WS/verl05_cuda 2>&1 | tail -1
python3 -m pip install --user -c $WS/cuda_constraints.txt \
  "tensordict==0.8.3" "ray[default]" hydra-core omegaconf codetiming dill pyarrow pandas pylatexenc \
  "transformers==4.49.0" accelerate "datasets==5.0.0" cachetools fastapi uvicorn pydantic math-verify wandb torchdata cloudpickle orjson peft 2> torchdata cloudpickle orjson 2>&11 | tail -3
python3 -c "import verl, tensordict, ray, transformers; print('verl', open('$WS/verl05_cuda/verl/version/version').read().strip(), '| transformers', transformers.__version__)"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$WS/data/gsm8k/train.parquet data.val_files=$WS/data/gsm8k/test.parquet \
  data.train_batch_size=16 data.max_prompt_length=256 data.max_response_length=128 \
  actor_rollout_ref.model.path=$WS/models/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.rollout.name=hf actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  algorithm.use_kl_in_reward=False ray_init.num_cpus=8 \
  trainer.logger=[console,wandb] trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=cuda \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=$STEPS \
  trainer.project_name=verl-gaudi-vs-a100 trainer.experiment_name=$EXP
echo "VERL_RC=$?"
