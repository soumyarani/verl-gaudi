#!/bin/bash
# Gaudi 1.5B LoRA-32 training (VARIANT=zenith|grpo): proven LoRA fix stack
# (KEEP_WRAP_POLICY + FSDP_PREFORWARD + lazy + chunked gen) at a viable base rate.
set -uo pipefail
WS=/scratch/ssamine4/verl_gaudi
RAYTMP=$RAY_TMPDIR
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
VARIANT=${VARIANT:?zenith|grpo}
echo "LORA15B $VARIANT: LAZY=${PT_HPU_LAZY_MODE:-unset}"
export VERL_KEEP_WRAP_POLICY=1 VERL_FSDP_PREFORWARD=1 VERL_LEN_PANELS=1
EXTRA=()
if [ "$VARIANT" = "zenith" ]; then
  export VERL_SIGN_ADVANTAGES=1 VERL_CS_HINGE=1
  export VERL_CS_MARGIN_POS=1.0 VERL_CS_MARGIN_NEG=0.0 VERL_CS_TERMINAL_EXEMPT=0
  EXTRA+=(algorithm.norm_adv_by_std_in_grpo=False actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_epochs=2)
else
  EXTRA+=(actor_rollout_ref.actor.ppo_mini_batch_size=32)
fi
python3.10 -m pip install --user --no-deps -e $WS/verl05 2>&1 | tail -1

run_verl() {
  python3.10 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$WS/data/gsm8k/train.parquet \
    data.val_files=$WS/data/gsm8k/test_200.parquet \
    data.train_batch_size=32 data.max_prompt_length=256 data.max_response_length=512 \
    actor_rollout_ref.model.path=$WS/models/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.strategy=fsdp +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer] \
    actor_rollout_ref.rollout.name=hf actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.micro_batch_size=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    ray_init.num_cpus=8 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=hinge-loss-grpo \
    trainer.experiment_name=gaudi-gsm8k-15b-lora32-$VARIANT \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.device=hpu \
    trainer.save_freq=-1 trainer.test_freq=10 \
    trainer.total_epochs=1 trainer.total_training_steps=40 "${EXTRA[@]}"
}

RC=1
ATT=/tmp/verl_attempt_$$.log
for attempt in 1 2 3; do
  echo "========== lora-eager attempt $attempt =========="
  run_verl 2>&1 | tee "$ATT"
  RC=${PIPESTATUS[0]}
  if [ "$RC" -eq 0 ]; then echo "attempt $attempt SUCCEEDED"; break; fi
  if grep -qE "GCS cannot find the node|node timed out during startup|metrics_agent_port|RPC error: Deadline|Timed out waiting for|raylet.*failed to startup|Device acquire failed|Device not found" "$ATT"; then
    clean_ray
  else
    echo "non-infra failure (RC=$RC), NOT retrying"; break
  fi
done
rm -f "$ATT" 2>/dev/null || true
echo "VERL_RC=$RC"
