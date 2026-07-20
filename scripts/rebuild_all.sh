#!/bin/bash
# Full env rebuild for the AIR Gaudi pod after a restart (emptyDir /workspace wiped).
# Reconstructs: base stack + all HPU patches + Qwen3-8B + LiveCodeBench data + SDPO reward.
# Run from /workspace after: git clone https://github.com/soumyarani/verl-gaudi /workspace/verl-gaudi-repo
set -e
REPO=/workspace/verl-gaudi-repo
FF=$REPO/patches/air/fixes_final
cd /workspace

echo "==== [1] base stack (venv + vllm + vllm-gaudi + verl0.9 + deps) ===="
bash $REPO/scripts/install_clean.sh
source /workspace/venv/bin/activate

echo "==== [2] apply HPU patches (complete files) ===="
cp $FF/hpu_model_runner.py                    /workspace/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py
cp $FF/detokenizer_utils.py                   /workspace/vllm/vllm/tokenizers/detokenizer_utils.py
cp $FF/vllm_rollout_utils.py                  /workspace/verl-src/verl/workers/rollout/vllm_rollout/utils.py
cp $FF/vllm_async_server.py                   /workspace/verl-src/verl/workers/rollout/vllm_rollout/vllm_async_server.py
cp $FF/bucketed_weight_transfer.py            /workspace/verl-src/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py
cp $FF/plugin_checkpoint_engine_hccl_hpu.py   /workspace/verl-src/verl/plugin/checkpoint_engine/hccl_hpu.py
cp $FF/single_controller_ray_base.py          /workspace/verl-src/verl/single_controller/ray/base.py
cp $FF/workers_rollout_replica.py             /workspace/verl-src/verl/workers/rollout/replica.py
cp $FF/workers_rollout_vllm_rollout.py        /workspace/verl-src/verl/workers/rollout/vllm_rollout/vllm_rollout.py
python $REPO/patches/air/patch_air_gsm8k_flexible.py || true
python -c "import ast,glob; [ast.parse(open(f).read()) for f in ['/workspace/verl-src/verl/plugin/checkpoint_engine/hccl_hpu.py']]; print('patches AST OK')"

echo "==== [3] Qwen3-8B + GSM8k(smoke) + LiveCodeBench data ===="
export HF_HOME=/workspace/hf
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-8B')"
mkdir -p /workspace/data
python $REPO/scripts/lcb/prep_lcb_v02.py

echo "==== [4] SDPO reward (rich-feedback code executor) ===="
[ -d /workspace/SDPO ] || git clone -q --depth 1 https://github.com/lasgroup/SDPO /workspace/SDPO
cp /workspace/SDPO/verl/utils/reward_score/feedback/code.py /workspace/lcb_code_reward.py
cp $REPO/scripts/lcb/lcb_reward.py /workspace/lcb_reward.py
cp $REPO/scripts/lcb/run_lcb_8b_lora_v01.sh /workspace/run_lcb_8b_lora_v01.sh
cd /root && VERL_PLATFORM=hpu python -c "import verl, vllm, vllm_gaudi; import sys; sys.path.insert(0,'/workspace'); import lcb_reward; print('REBUILD_OK: stack+reward import clean')"
echo "==== REBUILD_ALL_DONE ===="
