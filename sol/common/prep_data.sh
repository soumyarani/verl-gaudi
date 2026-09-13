#!/bin/bash
# Regenerate the GSM8k parquet files into $VG_HOME/data/gsm8k (idempotent).
# Only needed if $VG_HOME/data/gsm8k is missing (it is shipped in home by default).
# Produces train.parquet, test.parquet, and a small test_200.parquet used by LoRA smokes.
set -uo pipefail
VG_HOME="${VG_HOME:-$HOME/verl_gaudi}"
OUT="$VG_HOME/data/gsm8k"; mkdir -p "$OUT"

# Uses verl's bundled preprocessing; run inside whichever env has verl importable.
python3 "$VG_HOME/runtime/verl05/examples/data_preprocess/gsm8k.py" --local_dir "$OUT"

# Small held-out val used by the LoRA smoke (keeps val cheap).
python3 - "$OUT" <<'PY'
import sys, pandas as pd, os
out = sys.argv[1]
df = pd.read_parquet(os.path.join(out, "test.parquet"))
df.head(200).to_parquet(os.path.join(out, "test_200.parquet"))
print("wrote test_200.parquet:", len(df.head(200)), "rows")
PY
ls -la "$OUT"
