set -e
apt-get install -y -qq python3.10-venv >/dev/null 2>&1
python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate; pip install -q -U pip
cd /workspace
echo "[1/3] vllm_gaudi 0.24"
git clone https://github.com/vllm-project/vllm-gaudi >/dev/null 2>&1
cd vllm-gaudi; export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null); cd ..
git clone https://github.com/vllm-project/vllm >/dev/null 2>&1; cd vllm; git checkout $VLLM_COMMIT_HASH >/dev/null 2>&1
pip install -q -r <(sed '/^torch/d' requirements/build/cuda.txt) >/dev/null 2>&1
VLLM_TARGET_DEVICE=empty pip install -q --no-build-isolation -e . >/dev/null 2>&1
cd ../vllm-gaudi; pip install -q -e . >/dev/null 2>&1
echo "[2/3] verl 0.9 @1ff76cc + patch"
cd /workspace
git clone -q -b vllm-gaudi https://github.com/soumyarani/verl-gaudi verl-gaudi-repo >/dev/null 2>&1
git clone -q https://github.com/volcengine/verl.git verl-src >/dev/null 2>&1
cd verl-src; git checkout -q 1ff76cc625e9820d2434dad1b6d9b8e5dd26a359
git apply /workspace/verl-gaudi-repo/patches/verl09_main.diff && echo PATCH_OK || echo PATCH_FAIL
pip install -q -e . --no-deps >/dev/null 2>&1
echo "[3/3] deps"
pip install -q "ray[default]" tensordict codetiming hydra-core omegaconf pyarrow pandas datasets accelerate peft dill pybind11 numpy TransferQueue torchdata pylatexenc math-verify word2number wandb >/dev/null 2>&1
cd /root; python -c "import vllm,vllm_gaudi,verl,transfer_queue; print('STACK_OK')"
echo INSTALL_CLEAN_DONE
