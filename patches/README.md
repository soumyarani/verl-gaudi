# patches/ — index

Two kinds of change live here:
- **verl source** patches (`*.diff`, `platform_hpu.py`) — apply inside the relevant verl clone.
- **installed-library** patches (`env/patch_*.py`) — edit files in `site-packages`; **re-run after any reinstall**
  of that library (they are idempotent string-replacements).

See [`../docs/03-debugging-journey.md`](../docs/03-debugging-journey.md) for the *why* behind each, and
[`../docs/02-gaudi-port.md`](../docs/02-gaudi-port.md) for the concept grouping.

## verl source

| File | Applies to | What it does |
|------|-----------|--------------|
| `verl05.diff` | `verl05/` (verl **0.5.0**) | The **HF-rollout path that generated tokens**. HPU device registration (`device.py`), HPU Ray resource + placement bundle (`base.py`), HPU count in resource check (`ray_trainer.py`), `ray.init(resources={HPU:8})` (`main_ppo.py`), attn `flash_attention_2→sdpa`, FSDP pre-move to HPU, HFRollout fixes (`num_return_sequences=1`, inputs→HPU, drop bf16 autocast, lazy_mode, skip `summon_full_params`). |
| `verl09_main.diff` | `verl/` (verl **0.9.0.dev0**) | The **vLLM path**. Registers the platform plugin, HPU FSDP `EngineRegistry` entry (`device=[cuda,npu,hpu]`), HPU Ray resource in `base.py`, attn→sdpa, FSDP pre-move, model attn impl. |
| `platform_hpu.py` | drop into `verl/verl/plugin/platform/` (verl 0.9) | The new **`PlatformHPU`** backend: `device=hpu`, `torch.hpu`, `hccl`, `vendor=intel`, Ray resource `"HPU"`, visible-devices via `HABANA_VISIBLE_MODULES`, `empty_cache`/`StrEnum` shims, optimum-habana adapt hook. Registered by one import line appended to `platform_manager.py` (included in `verl09_main.diff`). |

Apply a diff:
```bash
cd /scratch/ssamine4/verl_gaudi/verl05 && git apply /path/to/patches/verl05.diff   # or: patch -p1 < ...
```

## env/ — installed-library patches (re-apply after reinstall)

| Script | Target library | What/why |
|--------|---------------|----------|
| `patch_ray.py` | Ray `util/scheduling_strategies` / `ray_option_utils.py` (PT image, ray 2.55) | Allow **fractional `HPU`** quantities (`if resource_name != "HPU" and ...`) so colocated roles can each request 1/3 HPU. |
| `patch_ray_vllm.py` | Ray `ray_option_utils.py` (vLLM image, ray 2.47) | Same fractional-HPU allowance for the vLLM-image Ray version. |
| `patch_strenum.py` | stdlib `enum` usage in vLLM deps | Shim `enum.StrEnum` for Python 3.10 (vLLM code assumes 3.11). |
| `patch_oh_eos.py` | optimum-habana `generation/utils.py` `_sample` | Fix a real OH bug: `torch.tensor(eos_token_id)` lands on **CPU** while inputs are on HPU → build it with `device=input_ids.device`. |
| `patch_oh_guard.py` | optimum-habana `_sample` | Guard the **empty eos-search slice** that triggers `argmax(): reduction dim 1 has non-zero size`. |
| `patch_oh_ver.py` | optimum-habana version check | No-op `check_synapse_version` so OH 1.18 runs against the 1.24 driver. |

Apply one:
```bash
PYTHONUSERBASE=/scratch/ssamine4/verl_gaudi/cpkgs /usr/bin/python3.10 patches/env/patch_oh_eos.py
```
(Each script locates its target in the active `site-packages` and rewrites in place; safe to run twice.)

## Order of application (HF path, the one that ran)
1. Build/enter the PT container; `PYTHONUSERBASE=$WS/cpkgs`.
2. `git apply verl05.diff` in `verl05/`.
3. Install `transformers==4.49`, `optimum-habana==1.18` (`scripts/install_oh18.sh`) under `cpkgs`.
4. `patch_oh_ver.py`, `patch_oh_eos.py`, `patch_oh_guard.py`, `patch_ray.py`.
5. `sbatch scripts/run_05.sh`.

## Session-2 patches (now folded into `verl05.diff`)

These were added in the "fresh-eyes" pass and are part of the regenerated `verl05.diff`:

| Where | Change | Why |
|-------|--------|-----|
| `verl/workers/fsdp_workers.py` `get_sharding_strategy` | return `ShardingStrategy.NO_SHARD` when `device_mesh.size()==1` | `FULL_SHARD` on one card emits `storage._resize_` ops HPU lazy mode can't do → native crash in `generate`. NO_SHARD keeps params resident. |
| `verl/utils/device.py` | coerce `ReduceOp.AVG/PREMUL_SUM → SUM` for `world_size==1` groups | Habana HCCL rejects AVG (`hccl_kernels.cpp:158`); on 1 rank the reduction is the identity so SUM is exact. Kept in `device.py` (verl-only) so it does NOT load `torch` in Ray helper processes. |
| `verl/trainer/main_ppo.py` ray.init | `include_dashboard=False` | minor; the raylet still waits for the metrics agent, so this alone does not fix the agent deadlock. |

Script changes (in `scripts/`):
- `_run_inside_05.sh`: **cache-warming Ray retry loop** — runs `main_ppo` up to 4×, `clean_ray()` between attempts
  (kill `gcs_server`/`raylet`/`dashboard`/`plasma`, wipe `RAY_TMPDIR`); only retries *infra* signatures
  (GCS/agent/Device-acquire), breaks on real training errors. Rides past the timing-sensitive metrics-agent deadlock.
- `run_05.sh`: `RAY_raylet_start_wait_time_s=600`; preserve `logs/raylogs_<JID>/{raylet,gcs_server,dashboard*}.*`
  on failure; `--exclude` stuck-HPU-module nodes.

> See [`../docs/03-debugging-journey.md`](../docs/03-debugging-journey.md) **Phase 7** for the full root-cause story,
> and [`../docs/memory.md`](../docs/memory.md) for current state + next levers.
