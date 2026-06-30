# skills.md — The Playbook (how to actually run this on Sol)

> Operational recipes + hard-won gotchas. Pair with `memory.md` (state) and `02-gaudi-port.md` (why).

## Working principles (carried from the project)
1. **Think before coding** — surface assumptions/tradeoffs; ask when genuinely blocked.
2. **Simplicity first** — minimal change that solves it.
3. **Surgical changes** — match style; don't refactor what isn't broken; every changed line traces to the goal.
4. **File locations & versioning** — don't write to scratch/tmp the user hasn't approved without asking; version
   experiment scripts `<name>_vNN.ext` (highest = latest).
5. **Goal-driven** — define a verify check per step; loop until verified.

## Cluster mechanics (the non-obvious stuff that cost hours)
- **`sbatch` only.** Interactive `srun`/`salloc` job-steps fail on Sol. Submit one-shot batch jobs, poll
  `squeue -j <id>`, read the log.
- **Run on a gaudi node.** The login node can't `import` HPU torch (old glibc). Allocate:
  `#SBATCH -p gaudi -G 1 -c 16 --mem=96G` (GPU jobs need >=24 GB; nodes are exclusive). Exclude flaky nodes:
  `#SBATCH --exclude=gaudi001,gaudi002,gaudi003`.
- **Long downloads = sbatch, not `nohup &`** (SSH close kills nohup). Build `.sif` with
  `APPTAINER_TMPDIR=/tmp` (node-local) — beegfs small-file extraction is ~30x slower.
- **VPN-only host.** If `sol` stops resolving, the ASU VPN dropped — reconnect it (cannot fix agent-side).
- **Quoting:** to ship a script to the cluster, write it locally and `cat localfile | ssh sol 'cat > remote'` —
  inline single-quotes inside a single-quoted SSH arg break things. Same for patches: write `patch_*.py` locally,
  `cat patch.py | ssh sol python3`.

## THE run recipe (Apptainer + HPU) — and why each flag exists
```bash
apptainer exec --cleanenv --no-home --bind /scratch:/scratch --bind /tmp:/tmp \
  --env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \   # graph compiler needs TPC kernels
  --env HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
  --env HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
  --env HABANA_LOGS=$WS/run/habana_logs.$SLURM_JOB_ID \          # /var/log is read-only -> SynapseAI aborts
  --env PYTHONUSERBASE=$WS/cpkgs \                               # writable user-site (container is read-only)
  --env PT_HPU_LAZY_MODE=0|1 \                                   # 0=eager (FSDP storage ops), 1=lazy (optimum-habana)
  --env RAY_agent_register_timeout_ms=300000 \                   # else raylet crashes waiting for the agent
  --env RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  "$SIF" /usr/bin/python3.10 ...                                 # container python; NOT host 3.12
```
Gotchas baked into the above:
- `--cleanenv --no-home`: stops Sol's Apptainer from binding host `~/.local` (a CUDA torch) and Lmod over the
  container. Without it `import torch` gets the wrong (CUDA) torch and `python3` is 3.12.
- Start Ray **in-process via verl** (don't use the `ray` CLI in the vLLM image — it's broken). Pass resources to
  `ray.init` through Hydra: `+ray_kwargs.ray_init.resources={HPU:8} +ray_kwargs.ray_init.num_gpus=0
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1 +ray_kwargs.ray_init._temp_dir=$RAYTMP`.

## Reproduce the two paths

### A) vLLM path (engine launches; blocked on colocation) — `scripts/run_vllm.sh`
- Container `gaudi_124_vllm.sif`, `PYTHONUSERBASE=$WS/cpkgs_vllm`.
- `rollout.name=vllm rollout.tensor_model_parallel_size=1 rollout.gpu_memory_utilization=0.4 rollout.enforce_eager=True`.
- env: `VLLM_SKIP_WARMUP=true PT_HPU_ENABLE_LAZY_COLLECTIVES=true`.
- Re-apply `env/patch_ray_vllm.py` (fractional HPU) + `env/patch_strenum.py` after any ray reinstall.

### B) HF path (generates tokens; catch-22) — `scripts/run_05.sh`
- Container `gaudi_124_pt210.sif`, `PYTHONUSERBASE=$WS/cpkgs`, **verl05** clone.
- `rollout.name=hf rollout.tensor_model_parallel_size=1 trainer.use_v1=False trainer.device=hpu`
  `+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]`.
- Needs `transformers==4.49 + optimum-habana==1.18` in `cpkgs` and `env/patch_oh_*.py` applied.
- `PT_HPU_LAZY_MODE=0` (eager) got furthest (generation ~36 s).

## Installing python deps into a read-only container
```bash
PYTHONUSERBASE=$WS/cpkgs  python3.10 -m pip install --user --no-deps -e $WS/verl     # editable verl
PYTHONUSERBASE=$WS/cpkgs  python3.10 -m pip install --user -c $WS/cconstraints.txt <pkgs>   # pin torch/numpy/pandas
```
- `cconstraints.txt` pins `torch==2.10.0a0+...`, `numpy`, `pandas` so deps can't downgrade the HPU torch.
- `ray[default]` (not bare `ray`) — the metrics agent needs aiohttp/prometheus or the raylet crashes.
- After reinstalling ray or optimum-habana, **re-run the matching `env/patch_*.py`** (those edit installed files).

## Verifying a green HPU before debugging higher layers
```python
import habana_frameworks.torch, torch, habana_frameworks.torch.core as htcore
x = torch.randn(64,64, device="hpu"); y=(x@x).sum(); htcore.mark_step(); print(float(y.cpu()))  # must print a number
```
If this fails with `synStatus 26`, your userspace doesn't match the driver — fix the env before anything else.

## Lazy vs eager cheat-sheet (HPU)
| Need | Mode | Why |
|------|------|-----|
| FSDP `summon_full_params` / `storage._resize_` | **eager** (`PT_HPU_LAZY_MODE=0`) | resize unsupported in lazy |
| optimum-habana static-shape `generate` | **lazy** (`=1`) | designed around `mark_step` per token |
| trivial tensor ops | either | both compile fine |
The conflict between rows 1 and 2 is the core HF-path wall.
