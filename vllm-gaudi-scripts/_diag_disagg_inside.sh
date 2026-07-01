#!/bin/bash
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
RAYTMP=$RAY_TMPDIR
PYSPY=$WS/cpkgs_vllm/bin/py-spy
DLOG=$WS/logs/disagg_diag_run.log; : > "$DLOG"
pkill -9 -f gcs_server 2>/dev/null || true; pkill -9 -f raylet 2>/dev/null || true
rm -rf /tmp/ray "$RAYTMP"/* 2>/dev/null || true; sleep 3
echo "=== DIAG: launching disaggregated run in background, will py-spy the hang ==="
python3.10 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$WS/data/gsm8k/train.parquet data.val_files=$WS/data/gsm8k/test.parquet \
  data.train_batch_size=16 data.max_prompt_length=256 data.max_response_length=128 \
  actor_rollout_ref.model.path=$WS/models/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.nnodes=1 actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.checkpoint_engine.backend=hccl \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  algorithm.use_kl_in_reward=False \
  ray_kwargs.ray_init.num_cpus=8 +ray_kwargs.ray_init.num_gpus=0 \
  '+ray_kwargs.ray_init.resources={HPU:8}' \
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1 +ray_kwargs.ray_init._temp_dir=$RAYTMP \
  trainer.logger=[console] trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=hpu \
  trainer.use_v1=True trainer.v1.trainer_mode=separate_async \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 trainer.total_training_steps=3 \
  trainer.project_name=verl_gaudi trainer.experiment_name=qwen05b_gsm8k_grpo_hpu_disagg > "$DLOG" 2>&1 &
RUNPID=$!
echo "run pid=$RUNPID; waiting for the hang marker..."
for i in $(seq 1 40); do
  grep -q "Platform override from VERL_PLATFORM: hpu" "$DLOG" 2>/dev/null && { echo "hang marker seen at iter $i"; break; }
  kill -0 $RUNPID 2>/dev/null || { echo "run exited early"; break; }
  sleep 15
done
echo "=== settling 150s into the hang ==="; sleep 150
echo "=== last 8 log lines ==="; tail -8 "$DLOG"
echo "=== python process tree ==="
ps -eo pid,ppid,etimes,stat,comm,args 2>/dev/null | grep -iE "python3.10|ray::|main_ppo|TaskRunner|transfer|EngineCore|raylet|gcs" | grep -v grep | head -40
echo "=== PY-SPY DUMPS (all python procs) ==="
for pid in $(pgrep -f "python3.10|ray::|TaskRunner|transfer_queue|EngineCore|main_ppo" 2>/dev/null | sort -u); do
  echo "----- py-spy dump pid $pid : $(ps -o args= -p $pid 2>/dev/null | cut -c1-90) -----"
  timeout 25 $PYSPY dump --pid $pid 2>&1 | head -45
  echo
done
kill -9 $RUNPID 2>/dev/null || true
pkill -9 -f main_ppo 2>/dev/null || true; pkill -9 -f raylet 2>/dev/null || true
echo "=== DIAG DONE ==="
