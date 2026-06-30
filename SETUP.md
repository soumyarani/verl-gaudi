# SETUP.md — Reproduce verl GRPO on Intel Gaudi 2, from zero

> Goal: get `verl` running a Qwen2.5‑0.5B GSM8k GRPO job on an Intel Gaudi 2 (HPU) node and complete a few training
> steps. This is the exact, tested recipe behind the **3/3-steps, `VERL_RC=0`** result (see
> [`docs/03-debugging-journey.md`](docs/03-debugging-journey.md) Phase 8).
>
> **Audience:** any engineer or AI agent. Every command is literal. Where a value is site-specific
> (`/scratch/<you>`, cluster name) it's called out.

---

## 0. What you need and the ground rules

- **Hardware:** an Intel Gaudi 2 (HL‑225) node with **SynapseAI driver 1.24** (`hl-smi` to check). This recipe was
  run on ASU **Sol**, `gaudi` partition, via SLURM.
- **Container runtime:** Apptainer/Singularity.
- **The hard rules that cost us days (don't skip):**
  - **Use batch jobs (`sbatch`), not interactive `srun`/`salloc`** — interactive job-steps fail on Sol.
  - **Do everything on a Gaudi compute node**, not the login node (login glibc is too old to import HPU torch).
  - **Match userspace to the driver.** A 1.22/1.23 SynapseAI userspace against a 1.24 driver fails even a trivial
    `(x@x).sum()` with `synStatus 26`. Use the **1.24** container below.
  - **Request the node `--exclusive`.** Sol provides *no* per-job HPU isolation (all 8 `/dev/accel*` are
    world-visible), so on a shared node your process grabs a module another job already holds →
    `synStatus=8 Device acquire failed`. Exclusive guarantees a free module.

Set a workspace once (everything lives here):
```bash
export WS=/scratch/$USER/verl_gaudi      # adjust to your scratch
mkdir -p $WS $WS/scripts $WS/logs $WS/run $WS/data $WS/models $WS/hf_cache $WS/cpkgs
```

---

## 1. Pull the SynapseAI 1.24 PyTorch container

```bash
# on a node with internet (a gaudi compute node), via sbatch or directly:
apptainer pull $WS/gaudi_124_pt210.sif \
  docker://vault.habana.ai/gaudi-docker/1.24.0/ubuntu22.04/habanalabs/pytorch-installer-2.10.0:1.24.0-1007
```
This image has Python 3.10 at `/usr/bin/python3.10` and a SynapseAI‑1.24‑matched `torch 2.10`.

Sanity-check the HPU is green (must print a number, no `synStatus 26`):
```bash
apptainer exec --cleanenv --no-home --bind /scratch \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  --env HABANA_LOGS=$WS/run/hl --env PT_HPU_LAZY_MODE=1 \
  $WS/gaudi_124_pt210.sif /usr/bin/python3.10 -c \
  'import habana_frameworks.torch, torch, habana_frameworks.torch.core as c; \
   x=torch.randn(64,64,device="hpu"); y=(x@x).sum(); c.mark_step(); print(float(y.cpu()))'
```

---

## 2. Get the Gaudi‑modified verl

This repo ships the **complete** modified verl 0.5.0 tree at [`verl05-gaudi/`](verl05-gaudi/). Copy it to your
workspace as the editable package:
```bash
cp -r verl05-gaudi $WS/verl05
```
(Equivalent: `git clone --branch v0.5.0 https://github.com/volcengine/verl $WS/verl05` then
`git apply patches/verl05.diff`. The diff and the full tree are kept in sync.)

What changed vs upstream (all in `patches/verl05.diff`, explained in [`docs/02-gaudi-port.md`](docs/02-gaudi-port.md)
and [`patches/README.md`](patches/README.md)):
- `verl/utils/device.py` — register `hpu`; optimum‑habana `adapt_transformers_to_gaudi()`; `empty_cache` shim;
  **coerce `ReduceOp.AVG/PREMUL_SUM → SUM`** on `world_size==1` (HCCL has no AVG).
- `verl/workers/fsdp_workers.py` — **`NO_SHARD` for a size‑1 mesh**; **`mixed_precision=None` on HPU**; pre‑move
  module to HPU before `FSDP(...)`; attn `flash_attention_2 → sdpa`.
- `verl/workers/rollout/hf_rollout.py` — `num_return_sequences=1`; inputs→HPU; drop bf16 autocast; lazy generate.
- `verl/trainer/main_ppo.py` — `ray.init(resources={"HPU":8}, num_gpus=0, _node_ip_address="127.0.0.1",
  include_dashboard=False)`.
- `verl/single_controller/ray/base.py`, `verl/trainer/ppo/ray_trainer.py` — schedule/count the `HPU` Ray resource.

---

## 3. Install Python deps into a writable user-site (`cpkgs`)

The container is read-only, so deps go to `PYTHONUSERBASE=$WS/cpkgs`. **Pin torch/numpy/pandas** so nothing
downgrades the HPU torch — `scripts/cconstraints.txt`:
```
torch==2.10.0a0+gitc1e5ed4
numpy==2.2.6
pandas==2.3.3
```
Install (run inside the container; see `scripts/setup_container_v17.sh`):
```bash
RUN="apptainer exec --cleanenv --no-home --bind /scratch --env PYTHONUSERBASE=$WS/cpkgs $WS/gaudi_124_pt210.sif"
$RUN python3.10 -m pip install --user --no-deps -e $WS/verl05
$RUN python3.10 -m pip install --user -c $WS/cconstraints.txt \
    "tensordict==0.8.3" "ray[default]==2.55.1" hydra-core omegaconf codetiming \
    dill pyarrow pandas pylatexenc "transformers==4.49.0" accelerate datasets \
    cachetools fastapi uvicorn pydantic math-verify
# Habana generate path needs a Habana-validated transformers/optimum-habana pin:
$RUN python3.10 -m pip install --user -c $WS/cconstraints.txt "transformers==4.49.0"
$RUN python3.10 -m pip install --user --no-deps "optimum-habana==1.18.0"
```
Notes:
- **`ray[default]`** (not bare `ray`) — the metrics agent needs aiohttp/prometheus or the raylet crashes.
- **transformers 4.49 + optimum-habana 1.18** is the combination whose `GaudiQwen2` generate works here.

---

## 4. Apply the installed-library patches (`patches/env/`)

These edit files *inside* `cpkgs` (Ray, optimum-habana). **Re-run them after any reinstall of those libs.**
```bash
for p in patch_ray.py patch_oh_ver.py patch_oh_eos.py patch_oh_guard.py; do
  $RUN python3.10 $WS/patches_env/$p     # copy patches/env/*.py to $WS/patches_env first
done
```
- `patch_ray.py` — allow **fractional `HPU`** resource quantities (colocated actor/rollout/ref each want 1/3 HPU).
- `patch_oh_ver.py` — no-op optimum-habana's SynapseAI version check (runs OH 1.18 on the 1.24 driver).
- `patch_oh_eos.py` — fix an OH bug: build the eos tensor on `input_ids.device` (not CPU).
- `patch_oh_guard.py` — guard an empty eos-search slice (`argmax` on a 0-dim).

---

## 5. Prepare data + model (`scripts/prep_data_v20.sh`)

```bash
$RUN python3.10 $WS/verl05/examples/data_preprocess/gsm8k.py --local_save_dir $WS/data/gsm8k
$RUN python3.10 -c 'from huggingface_hub import snapshot_download as d; \
  d("Qwen/Qwen2.5-0.5B-Instruct", local_dir="'$WS'/models/Qwen2.5-0.5B-Instruct")'
```

---

## 6. Run it (`scripts/run_05.sh` + `scripts/_run_inside_05.sh`)

Submit the batch job (the wrapper requests `-p gaudi -G 1 -c 16 --mem=96G --exclusive` and forwards the HPU env):
```bash
sbatch scripts/run_05.sh           # or scripts/bench_gaudi_v01.sh for the 100-step benchmark
```
The wrapper's apptainer invocation — **every env var matters**:
```bash
apptainer exec --cleanenv --no-home --bind /scratch --bind /tmp \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \      # TPC kernels for the graph compiler
  --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  --env HABANA_LOGS=$WS/run/hl.$SLURM_JOB_ID \                       # /var/log is read-only → set this
  --env PYTHONUSERBASE=$WS/cpkgs \
  --env PT_HPU_LAZY_MODE=1 --env PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
  --env VERL_PLATFORM=hpu \
  --env TMPDIR=$RAYTMP --env RAY_TMPDIR=$RAYTMP \                     # per-job Ray temp on node-local /tmp
  --env RAY_agent_register_timeout_ms=300000 \
  --env RAY_raylet_start_wait_time_s=150 \
  $WS/gaudi_124_pt210.sif bash $WS/scripts/_run_inside_05.sh
```
The inside script (`_run_inside_05.sh`) runs `python3.10 -m verl.trainer.main_ppo ...` with these key overrides:
```
algorithm.adv_estimator=grpo
actor_rollout_ref.model.path=$WS/models/Qwen2.5-0.5B-Instruct
actor_rollout_ref.rollout.name=hf                 # HF rollout (single-card train+generate; vLLM colocation is blocked on Gaudi)
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.n=4
actor_rollout_ref.actor.strategy=fsdp
+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]
trainer.device=hpu trainer.n_gpus_per_node=1 trainer.nnodes=1
trainer.use_v1=False                              # classic trainer (avoids transfer_queue)
trainer.total_training_steps=3                    # or 100 for the benchmark
data.train_batch_size=16 data.max_prompt_length=256 data.max_response_length=128
actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=8 ...
```
and wraps that call in a **cache-warming retry loop**: if Ray fails to start (its metrics-agent startup is
timing-flaky in containers), it cleans stale Ray and retries — the page cache is warm on the retry so the agent
wins the race. Only *infra* failures retry; real training errors stop immediately.

**Success looks like:** in `slurm-<JID>.out` you'll see `After FSDP` → `Training Progress: N/M` →
`step:K - ... actor/pg_loss ... actor/grad_norm ...` → `VERL_RC=0`. On failure, the wrapper preserves
`logs/raylogs_<JID>/raylet.err` — read it for the real cause.

---

## 7. Troubleshooting (ctrl-F your error)

| Error | Cause → Fix |
|-------|-------------|
| `synStatus 26` on a trivial op | userspace ≠ driver → use the **1.24 container** |
| `GLIBC_2.33 not found` on import torch | you're on the **login node** → run on a gaudi node |
| `synStatus=8 Device acquire failed / Device not found` | shared node, module taken → **`--exclusive`** |
| `GCS cannot find the node` / `Timed out waiting for metrics_agent_port` | Ray agent startup deadlock → the **retry loop** rides past it; read `logs/raylogs_<JID>/raylet.err` |
| `This function should not be called in lazy flow` (`_resize_`) | FSDP storage free in lazy mode → **`NO_SHARD`** (size-1 mesh) and **`mixed_precision=None`** on HPU (both already patched) |
| `Unsupported ReduceOp for HCCL` | HCCL has no AVG → **`AVG→SUM`** coercion in `device.py` (patched) |
| `int('all')` / `invalid literal ... 'all'` | don't set `HABANA_VISIBLE_MODULES=all`; it takes integers, or leave it unset |
| `Could not find the transformer layer class to wrap` | pass `wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]` (a **list**) |
| `No module named 'transfer_queue'` | `trainer.use_v1=False` |

Deeper "why" for every one of these: [`docs/03-debugging-journey.md`](docs/03-debugging-journey.md).
