# verl on ASU Sol — runnable setup for A100 + Gaudi (full + LoRA)

Agent-facing runbook. Everything here is **already installed on Sol** and **runs from
`$HOME/verl_gaudi`** so it survives scratch purges. Pick a backend, pick full or LoRA, `sbatch`.

```
ssh sol                       # ASU VPN required; Duo prompt on connect
cd ~/verl_gaudi
```

## TL;DR — one command per experiment

| Goal | Command |
|------|---------|
| Gaudi, full-param GRPO | `sbatch sol/gaudi/full_grpo.sbatch` |
| Gaudi, LoRA GRPO       | `sbatch sol/gaudi/lora_grpo.sbatch` |
| A100, full-param GRPO  | `sbatch sol/a100/full_grpo.sbatch` |
| A100, LoRA GRPO        | `sbatch sol/a100/lora_grpo.sbatch` |

Default = Qwen2.5-0.5B on GSM8k, a few steps, metrics to **wandb** (project `verl_gaudi`)
+ console. All four are verified to reach `VERL_RC=0`.

## Tuning without editing scripts (env overrides)

`sbatch` forwards your exported environment (`--export=ALL` is the default), so:

```bash
# bigger model / more steps (Gaudi LoRA, rank 64)
LORA_RANK=64 LORA_ALPHA=64 STEPS=40 \
  MODEL=$HOME/verl_gaudi/models/Qwen2.5-1.5B-Instruct \
  sbatch sol/gaudi/lora_grpo.sbatch

# A100 full run at the heavy benchmark config
STEPS=100 TBS=1024 MINI=256 RESP=1024 sbatch sol/a100/full_grpo.sbatch
```

Common knobs: `MODEL` (abs path under `~/verl_gaudi/models`), `STEPS`, `EXP` (wandb run name),
`LORA_RANK`/`LORA_ALPHA` (LoRA only), and on A100 `TBS`/`MINI`/`RESP` (batch/mini/resp-len).

## Where things live

```
~/verl_gaudi/                         # VG_HOME — durable, in $HOME (Isilon)
  sol/gaudi/*.sbatch  sol/a100/*.sbatch   the runnable scripts (this repo's sol/)
  containers/gaudi_124_pt210.sif          Gaudi SynapseAI-1.24 container (HF-rollout path)
  runtime/cpkgs/                          Gaudi user-site deps (transformers, optimum-habana)
  runtime/verl05/                         HPU-ported verl 0.5.0 (device=hpu path)
  runtime/verl_cuda/                      verl for the A100 CUDA path
  models/Qwen2.5-0.5B-Instruct  models/Qwen2.5-1.5B-Instruct
  data/gsm8k/{train,test,test_200}.parquet
  .wandb_key                              wandb api key (gitignored, NEVER commit)
$VG_WORK = /scratch/$USER/verl_gaudi_work # throwaway: outputs, ray tmp, A100 venv, hf cache
```

Scratch (`$VG_WORK`) is regenerable. If it is wiped: the Gaudi path needs nothing (all inputs
are in home); the A100 path rebuilds its vLLM venv automatically on next run (`setup_venv.sh`).

## The two backends differ — this matters

| | **Gaudi (HPU)** | **A100 (CUDA)** |
|--|--|--|
| Partition | `-p gaudi` (10× 8-chip HL-225) | `-p public --gres=gpu:a100:1` |
| Launch | Apptainer container (`--cleanenv --no-home`) | native uv venv on the node |
| Rollout | `hf` (in-process generate) | `vllm` (native, fast, **learns**) |
| Device | `trainer.device=hpu`, `VERL_PLATFORM=hpu` | `trainer.device=cuda` |
| verl | `runtime/verl05` (HPU port) | `runtime/verl_cuda` |

**Gaudi-specific fixes baked into the scripts** (do not remove — each was a real blocker):
1. `--exclusive` node — Sol gives no per-job HPU isolation; a shared node → `Device acquire failed`.
2. Ray retry loop — rides past the raylet↔metrics-agent startup deadlock (`RAY_agent_register_timeout_ms=300000`).
3. `NO_SHARD` + `mixed_precision=None` for the size-1 FSDP mesh (avoids lazy `storage._resize_` crash) — in the verl05 port.
4. `ReduceOp.AVG→SUM` coercion (HCCL rejects AVG; world_size==1 makes it identity) — in the port.
5. LoRA only: `VERL_KEEP_WRAP_POLICY=1` + `+...wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]`
   (verl05 otherwise nullifies the LoRA-aware wrap policy → mixed-`requires_grad` FSDP group crash).

## Checking a run

```bash
squeue -u $USER                                  # is it running / queued?
tail -f $VG_WORK/run/*/…  ||  ls slurm-*.out      # slurm-<jobid>.out in the submit dir
grep -E "step:|VERL_RC=|reward" slurm-<jobid>.out # steps + final return code
```
`VERL_RC=0` = success. Metrics also land in wandb (`verl_gaudi`).

Note: the 0.5B model scores ~0 reward on GSM8k in a handful of steps (too weak / cold start,
`grad_norm` can be `nan`) — that is a *training-quality* caveat, **not** a pipeline failure. Use
the A100 vLLM path with more steps / a bigger model to see reward actually climb.

## Rebuilding from scratch (if home artifacts are lost)

- Gaudi container: `scripts/setup_container_v17.sh` (builds `gaudi_124_pt210.sif` from the
  Habana 1.24 base). A100 venv: `sol/a100/setup_venv.sh`.
- Models: re-download Qwen2.5-0.5B/1.5B-Instruct into `~/verl_gaudi/models/`.
- Data: `bash sol/common/prep_data.sh`.

## Not on Sol

The **8B LoRA LiveCodeBench** run lives on the **AIR Gaudi Kubernetes** cluster (verl 0.9
disaggregated, `worktree-air-gaudi-grpo`), not here — its 8B weights (~16 GB) do not fit in
the Sol home quota. Ask before attempting to port it to Sol.
