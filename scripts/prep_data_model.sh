set -x
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf
mkdir -p /workspace/data
# gsm8k parquet via verl preprocess
python /workspace/verl-src/examples/data_preprocess/gsm8k.py --local_dir /workspace/data/gsm8k 2>&1 | tail -3
ls -la /workspace/data/gsm8k/
# pre-download the model
python -c "from huggingface_hub import snapshot_download; p=snapshot_download(\"Qwen/Qwen2.5-0.5B-Instruct\"); print(\"MODEL_AT\", p)" 2>&1 | tail -2
echo PREP_DONE
