#!/bin/bash
# env_common.sh — shared Apptainer launcher for verl-on-Gaudi (ASU Sol).
# Source this from an sbatch wrapper, then call:  vg_gaudi_run <inside_script.sh>
#
# Everything needed to RUN lives under $VG_HOME (durable home, survives scratch purge).
# Scratch is used only for regenerable outputs/caches under $VG_WORK.
#
# Overridable via environment:
#   VG_HOME   (default ~/verl_gaudi)                 code + container + cpkgs + models + data
#   VG_WORK   (default /scratch/$USER/verl_gaudi_work) outputs, ray tmp, hf cache (throwaway)
#   SIF       (default $VG_HOME/containers/gaudi_124_pt210.sif)
set -uo pipefail

VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
VG_WORK="${VG_WORK:-/scratch/$USER/verl_gaudi_work}"
SIF="${SIF:-$VG_HOME/containers/gaudi_124_pt210.sif}"
mkdir -p "$VG_WORK"/{run,logs,hf_cache}

vg_gaudi_run() {
  local INSIDE="$1"
  local JOB="${SLURM_JOB_ID:-manual$$}"
  local HLOG="$VG_WORK/run/habana_logs.$JOB"; mkdir -p "$HLOG"
  local RAYTMP="/tmp/verl_ray_$JOB";        mkdir -p "$RAYTMP"
  local WKEY=""; [ -f "$VG_HOME/.wandb_key" ] && WKEY="$(cat "$VG_HOME/.wandb_key")"

  # Forward experiment knobs ONLY when the caller set them, so each inside script keeps
  # its own default (full=3 steps, lora=5). Injecting defaults here would shadow those.
  local -a XENV=()
  [ -n "${MODEL:-}" ]      && XENV+=(--env "MODEL=$MODEL")
  [ -n "${STEPS:-}" ]      && XENV+=(--env "STEPS=$STEPS")
  [ -n "${EXP:-}" ]        && XENV+=(--env "EXP=$EXP")
  [ -n "${LORA_RANK:-}" ]  && XENV+=(--env "LORA_RANK=$LORA_RANK")
  [ -n "${LORA_ALPHA:-}" ] && XENV+=(--env "LORA_ALPHA=$LORA_ALPHA")

  echo "NODE=$(hostname) VG_HOME=$VG_HOME VG_WORK=$VG_WORK SIF=$SIF $(date)"
  echo "SLURM HPU env: MODULES=[${HABANA_VISIBLE_MODULES:-unset}] DEVICES=[${HABANA_VISIBLE_DEVICES:-unset}]"

  # --cleanenv --no-home is critical: host ~/.local CUDA torch + Lmod would shadow the container.
  apptainer exec --cleanenv --no-home \
    --bind "$VG_HOME:$VG_HOME" --bind "$VG_WORK:$VG_WORK" --bind /tmp:/tmp \
    --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
    --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
    --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
    --env HABANA_LOGS="$HLOG" \
    --env PYTHONUSERBASE="$VG_HOME/runtime/cpkgs" \
    --env HF_HOME="$VG_WORK/hf_cache" --env HF_HUB_OFFLINE=1 \
    --env PT_HPU_LAZY_MODE=1 --env PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
    --env VERL_PLATFORM=hpu --env TMPDIR="$RAYTMP" --env RAY_TMPDIR="$RAYTMP" \
    --env RAY_agent_register_timeout_ms=300000 \
    --env RAY_raylet_start_wait_time_s=150 \
    --env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    ${WKEY:+--env WANDB_API_KEY="$WKEY"} --env WANDB_DIR="$VG_WORK" --env WANDB__SERVICE_WAIT=300 \
    --env VG_HOME="$VG_HOME" --env VG_WORK="$VG_WORK" \
    "${XENV[@]}" \
    --env PATH="$VG_HOME/runtime/cpkgs/bin:/usr/local/bin:/usr/bin:/bin" \
    "$SIF" bash "$INSIDE"
  local rc=$?

  local SL="$VG_WORK/logs/raylogs_$JOB"; mkdir -p "$SL"
  cp -f "$RAYTMP"/session_latest/logs/raylet.*     "$SL/" 2>/dev/null || true
  cp -f "$RAYTMP"/session_latest/logs/gcs_server.* "$SL/" 2>/dev/null || true
  rm -rf "$RAYTMP"
  return $rc
}
