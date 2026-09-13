#!/bin/bash
# Runs INSIDE the Gaudi container. Full-parameter GRPO, Qwen2.5 on GSM8k, HF rollout.
# Proven: job 57954952 -> 3/3 steps, VERL_RC=0 (see docs/03-debugging-journey.md).
set -uo pipefail
: "${VG_HOME:?}" ; : "${VG_WORK:?}"
MODEL="${MODEL:-$VG_HOME/models/Qwen2.5-0.5B-Instruct}"
STEPS="${STEPS:-3}"
EXP="${EXP:-gaudi_full_grpo_$(basename "$MODEL")}"
RAYTMP="$RAY_TMPDIR"

clean_ray() {
  pkill -9 -f gcs_server 2>/dev/null || true
  pkill -9 -f raylet 2>/dev/null || true
  pkill -9 -f "ray/dashboard" 2>/dev/null || true
  pkill -9 -f "ray._private" 2>/dev/null || true
  pkill -9 -f plasma 2>/dev/null || true
  rm -rf /tmp/ray 2>/dev/null || true
  rm -rf "$RAYTMP"/* 2>/dev/null || true
  sleep 4
}
clean_ray
rm -rf /tmp/verl_ray_* 2>/dev/null || true

# Point the editable install at the HPU-ported verl 0.5.0 clone.
python3.10 -m pip install --user --no-deps -e "$VG_HOME/runtime/verl05" 2>&1 | tail -1
python3.10 -c "import verl; print('verl', open('$VG_HOME/runtime/verl05/verl/version/version').read().strip())"

run_verl() {
  python3.10 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$VG_HOME/data/gsm8k/train.parquet" \
    data.val_files="$VG_HOME/data/gsm8k/test.parquet" \
    data.train_batch_size=16 data.max_prompt_length=256 data.max_response_length=128 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.strategy=fsdp \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer] \
    actor_rollout_ref.rollout.name=hf actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    ray_init.num_cpus=8 \
    trainer.logger='["console","wandb"]' trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=hpu \
    trainer.save_freq=-1 trainer.test_freq=-1 \
    trainer.total_epochs=1 trainer.total_training_steps="$STEPS" \
    trainer.project_name=verl_gaudi trainer.experiment_name="$EXP"
}

# Ray on Sol occasionally loses the raylet<->metrics-agent handshake at startup; retry
# only transient infra failures, never a real training error.
RC=1; ATT=/tmp/verl_attempt_$$.log
for attempt in 1 2 3 4; do
  echo "========== gaudi full-grpo attempt $attempt =========="
  run_verl 2>&1 | tee "$ATT"; RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] && { echo "attempt $attempt SUCCEEDED"; break; }
  if grep -qE "GCS cannot find the node|node timed out during startup|metrics_agent_port|RPC error: Deadline|Timed out waiting for|raylet.*failed to startup|Device acquire failed|Device not found" "$ATT"; then
    echo "attempt $attempt: transient infra failure (RC=$RC), cleaning Ray + retrying"; clean_ray
  else
    echo "attempt $attempt: non-infra failure (RC=$RC), NOT retrying"; break
  fi
done
rm -f "$ATT" 2>/dev/null || true
echo "VERL_RC=$RC"
