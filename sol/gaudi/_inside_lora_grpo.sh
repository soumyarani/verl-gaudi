#!/bin/bash
# Runs INSIDE the Gaudi container. LoRA GRPO, Qwen2.5 on GSM8k, HF rollout.
# Proven: job 60999686 (lorasmoke3) -> 5/5 steps, VERL_RC=0.
#
# LoRA-on-Gaudi gotcha (see docs): verl05 nullifies the LoRA-aware FSDP wrap policy under
# HF rollout, producing a flat param group with MIXED requires_grad -> crash. Keeping the
# wrap policy (VERL_KEEP_WRAP_POLICY=1) makes groups uniform so default use_orig_params=False
# works and lazy mode (the proven generation path) is preserved.
set -uo pipefail
: "${VG_HOME:?}" ; : "${VG_WORK:?}"
MODEL="${MODEL:-$VG_HOME/models/Qwen2.5-0.5B-Instruct}"
STEPS="${STEPS:-5}"
LORA_RANK="${LORA_RANK:-32}"; LORA_ALPHA="${LORA_ALPHA:-32}"
EXP="${EXP:-gaudi_lora${LORA_RANK}_grpo_$(basename "$MODEL")}"
RAYTMP="$RAY_TMPDIR"

export VERL_KEEP_WRAP_POLICY=1
export VERL_FSDP_PREFORWARD=1

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

echo "LORA GRPO: rank=$LORA_RANK alpha=$LORA_ALPHA model=$MODEL steps=$STEPS"
python3.10 -m pip install --user --no-deps -e "$VG_HOME/runtime/verl05" 2>&1 | tail -1

run_verl() {
  python3.10 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$VG_HOME/data/gsm8k/train.parquet" \
    data.val_files="$VG_HOME/data/gsm8k/test_200.parquet" \
    data.train_batch_size=16 data.max_prompt_length=256 data.max_response_length=128 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.strategy=fsdp \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer] \
    actor_rollout_ref.rollout.name=hf actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.micro_batch_size=8 \
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

RC=1; ATT=/tmp/verl_attempt_$$.log
for attempt in 1 2 3; do
  echo "========== gaudi lora-grpo attempt $attempt =========="
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
