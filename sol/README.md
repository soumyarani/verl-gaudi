# `sol/` — clean, runnable verl setup on ASU Sol

Two backends, each with **full-parameter** and **LoRA** GRPO, parameterized for experiments.

- **Read [`AGENTS.md`](AGENTS.md)** for how to run anything (one `sbatch` per experiment).
- On the cluster this tree is deployed at `~/verl_gaudi/sol/` and runs from `$HOME` so it
  survives scratch purges.

```
sol/
  AGENTS.md              # the runbook — start here
  common/prep_data.sh    # regenerate GSM8k parquet (only if data/ is lost)
  gaudi/                 # HPU path: Apptainer container + HF rollout, device=hpu
    full_grpo.sbatch  lora_grpo.sbatch
    _inside_full_grpo.sh  _inside_lora_grpo.sh  env_common.sh
  a100/                  # CUDA path: native uv venv + vLLM rollout, device=cuda
    full_grpo.sbatch  lora_grpo.sbatch
    _inside_full_grpo.sh  _inside_lora_grpo.sh  setup_venv.sh
```

Verified: `sol/gaudi/lora_grpo.sbatch` re-run from `~/verl_gaudi` → `VERL_RC=0`. The other
three are adapted from proven runs (Gaudi `run_05`, A100 benchmark) — see AGENTS.md for exact status.
See the repo root `docs/` for the full Gaudi port, debugging journey, and A100-vs-Gaudi benchmark.
