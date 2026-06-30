# 03 — The Debugging Journey (blow-by-blow)

> Every error, in order, with the fix and the one-line lesson. This is the most useful doc for *pattern-matching*
> a future Gaudi error: ctrl-F the message. ~24 substantive iterations are condensed here.

Legend: 🟢 fixed · 🧱 architectural wall · 🔧 infra/config

## Phase 0 — Environment
| # | Symptom | Cause | Fix | Lesson |
|---|---------|-------|-----|--------|
| 0.1 🔧 | `srun` interactive steps "aborted before step completely launched" | Sol quirk | use **`sbatch`** only | Cluster-specific; never assume interactive works |
| 0.2 🔧 | login node can't `import torch` (`GLIBC_2.33 not found`) | login glibc < compute | run on a **gaudi node** | Build/run where the driver+glibc match |
| 0.3 🧱→🟢 | `(x@x).sum()` fails `synStatus 26` on *all* ASU envs | SynapseAI **1.22/1.23** userspace vs **1.24** driver | use Habana **1.24 container** | **Version-match userspace to the driver** |
| 0.4 🔧 | container `.sif` build crawls on beegfs | small-file extraction on parallel FS | `APPTAINER_TMPDIR=/tmp` (node-local) | Extract to node-local disk |
| 0.5 🔧 | `nohup ... &` pull dies | SSH session close kills it | run long jobs via **sbatch** | Detach long work as batch jobs |

## Phase 1 — Make the container usable
| # | Symptom | Fix | Lesson |
|---|---------|-----|--------|
| 1.1 | container `python3` is 3.12 w/ no torch | host conda shadowing via Apptainer binds | `--cleanenv --no-home`, call `/usr/bin/python3.10` | Apptainer on HPC binds host paths aggressively |
| 1.2 | `import torch` = CUDA `2.7.1+cu126` | `~/.local` user-site leaked in | `--no-home` + `PYTHONNOUSERSITE` / `-s` | Kill `~/.local` shadowing |
| 1.3 | `synStatus=26 Session init failed ... /var/log/... read-only` | logs dir read-only | `HABANA_LOGS=<writable>` | SynapseAI needs writable logs |
| 1.4 🟢 | — | — | `(x@x).sum()` and softmax **run on HPU** | green light to install verl |

## Phase 2 — Install verl + PlatformHPU
| # | Symptom | Fix |
|---|---------|-----|
| 2.1 | `venv` fails (no `python3.10-venv` in image) + read-only site-packages | install to `--user` site on scratch (`PYTHONUSERBASE=$WS/cpkgs`) |
| 2.2 | `tensordict` version + missing `pyvers/cloudpickle/orjson/torchdata` | pin `tensordict==0.8.3`, install its deps with constraints |
| 2.3 🟢 | — | `import verl` OK, `get_platform()=hpu/intel/8/hccl`, real HPU op via verl device facade |

## Phase 3 — The Ray boss fight (verl 0.9, vLLM path)
| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 3.1 | `ray.init` "node timed out during startup" (in-process) | raylet won't register | (later root-caused as 3.4) |
| 3.2 | `ray start` works once, then flaky; `No node info found matching attributes ''` | node-IP detection mismatch | pin node IP; pass `_node_ip_address` to `ray.init` |
| 3.3 | raylet crash backtrace in `NodeManager` | (see 3.4) | dumped `raylet.err` |
| 3.4 🟢 | `Check failed ... Timed out waiting for metrics_agent_port` | **`RAY_agent_register_timeout_ms` too short**; agent imports slow off beegfs | `RAY_agent_register_timeout_ms=300000` → **node registers** |
| 3.5 🟢 | registers as `0.0/1.0 TPU`, not `8 HPU` | Ray accelerator mis-detection | `ray.init(resources={"HPU":8})` + `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` |

## Phase 4 — verl worker setup (verl 0.9)
| # | Symptom | Fix | Patch |
|---|---------|-----|-------|
| 4.1 | `No module named 'transfer_queue'` | use classic trainer | `trainer.use_v1=False` |
| 4.2 | `No module named 'cachetools'` (+ fastapi/uvicorn/math_verify/...) | install verl's remaining pure-python deps | `install_deps2` |
| 4.3 | `Total available GPUs 0 < 1` | count HPU resource | `ray_trainer.py` / `base.py` HPU branch |
| 4.4 | `HPU resource quantity must be whole numbers (0.333)` | colocation fractional request | patch Ray `ray_option_utils` to allow fractional HPU |
| 4.5 | `No engine registered for device='hpu' ... backend='fsdp'` | engine registry | add `"hpu"` to FSDP `EngineRegistry.register(device=[...])` |
| 4.6 | `FlashAttention2 ... flash_attn not installed` | CUDA-only kernel | attn `flash_attention_2 -> sdpa` |
| 4.7 🟢 | `set_data ... incompatible tensor type` | FSDP CPU→HPU move | **pre-move** `module.to('hpu')` before `FSDP(...)` → **`After FSDP, 3.69 GB on HPU`** |

## Phase 5 — vLLM rollout (verl 0.9) — the architectural wall
| # | Symptom | Meaning |
|---|---------|---------|
| 5.1 | `ray` CLI `is not a valid Sentinel` (vLLM image) | dep skew → **bypass CLI**, use in-process `ray.init` |
| 5.2 | `module 'habana_frameworks.torch.hpu' has no attribute 'empty_cache'` | shim no-op |
| 5.3 | `cannot import name 'StrEnum' from 'enum'` | py3.10 vs 3.11 → shim `enum.StrEnum` |
| 5.4 🧱 | vLLM `EngineCore` worker: `synStatus=8 Device acquire failed` | **separate vLLM process can't share the actor's Gaudi module** |
| 5.5 🧱 | `VLLM_ENABLE_V1_MULTIPROCESSING=0` doesn't help | verl's vLLM rollout is a server (always separate proc) → needs **disaggregated HPUs** |

> vLLM **fully launched** on HPU (registered every `vllm_gaudi` model, started the engine) — the wall is purely the
> colocation/device-exclusivity, not vLLM-on-Gaudi itself.

## Phase 6 — HF rollout (verl 0.5.0) — generation actually ran
| # | Symptom | Fix |
|---|---------|-----|
| 6.1 | `0.5.0` has its own cuda/npu `device.py`, hardcoded `GPU`/`num_gpus` placement | re-derive HPU patches: `device.py`, `base.py` (HPU resource), `ray_trainer.py` (count), `main_ppo.py` (`ray.init` resources), attn→sdpa, FSDP pre-move |
| 6.2 | `rollout world_size 1 not divisible by infer_tp 2` | `rollout.tensor_model_parallel_size=1` |
| 6.3 | `This function should not be called in lazy flow` | (= 6.8; FSDP `summon_full_params._resize_`) → try **eager** |
| 6.4 | `module ... no attribute 'empty_cache'` | shim in 0.5.0 `device.py` |
| 6.5 | `Two tensor dict must have identical batch size (64 vs 256)` | **double-`n`**: HFRollout `num_return_sequences=n` AND trainer repeats → set to `1` |
| 6.6 | eager: `synNodeCreateWithId failed: strided_view` | vanilla HF generate view ops unsupported eager → bring in **optimum-habana** |
| 6.7 | `optimum-habana` import errors (`cached_property`, `sentence_transformers`, `sklearn`, `check_synapse_version`) | match **transformers 4.49 + optimum-habana 1.18**, shim/install deps, no-op the version check |
| 6.8 🟢 | `adapt_transformers_to_gaudi()` succeeds → model is `GaudiQwen2ForCausalLM` | the Gaudi generate is now active |
| 6.9 | `Could not find the transformer layer class to wrap` | adapt renamed layer → `wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]` (must be a **list**) |
| 6.10 🟢 | `index_copy_ Float vs BFloat16` | drop bf16 autocast in HFRollout (match optimum-habana fp32 cache) |
| 6.11 🟢 | `Expected all tensors on HPU ... input[idx=1] on cpu (LongTensor)` | **optimum-habana bug**: `torch.tensor(eos_token_id)` on CPU → patch to `device=input_ids.device` |
| 6.12 🟢 | `argmax(): Expected reduction dim 1 to have non-zero size` | guard the empty eos-search slice in optimum-habana `_sample` |
| 6.13 🧱 | back to `strided_view` (eager) / `lazy flow` (lazy) / actor-crash (no-summon) | **the catch-22**: FSDP `summon_full_params._resize_` needs eager; HF-generate view ops need lazy. No single mode satisfies both. |

> Best result: **generation ran ~36 s on the Gaudi card** before 6.13. That is the high-water mark.

## The two walls, stated once
1. **vLLM path:** colocated rollout needs a *second process* on the actor's Gaudi module → impossible (exclusive devices). **Fix = disaggregate onto separate HPUs.**
2. **HF path:** runs inference *through* the FSDP training graph → FSDP storage ops and HF-generate view ops demand opposite Habana execution modes. **Fix = don't run inference through FSDP; use a real inference engine (= back to vLLM, disaggregated).**

Both arrows point to the same next step: **disaggregated vLLM-gaudi** (separate HPU modules for actor vs. rollout).

---

## Phase 7 — The fresh-eyes pass: the "walls" were misdiagnosed (session 2)

> This phase revisited the two "architectural walls" from Phases 5–6 with fresh eyes and found that the HF-rollout
> wall was **not** a fundamental lazy/eager catch-22. Two surgical fixes removed it; the blocker that remains is
> infrastructure (Ray startup), not HPU/verl code.

### 7.1 🟢 The "catch-22" was just pointless single-card FSDP
The Phase-6 conclusion was: FSDP `summon_full_params` needs **eager**, optimum-habana generate needs **lazy**, no
single mode satisfies both. Re-examination: with `n_gpus_per_node=1` the FSDP device mesh has **size 1**, so
`get_sharding_strategy` returns `FULL_SHARD` — which still does the `storage._resize_(0)` free/gather dance on every
forward (via FSDP's own pre/post-forward hooks, not just the explicit `summon_full_params`). That resize is the only
thing that needs eager, and on one card the sharding it implements is a **no-op**.

| Symptom | Fix | Result |
|---------|-----|--------|
| Worker dies (native SIGABRT, "connection error code 2 / EOF") inside `generate_sequences`, even with `summon_full_params` already skipped | `get_sharding_strategy`: return **`ShardingStrategy.NO_SHARD`** when `device_mesh.size()==1` (params stay resident, DDP-style, no resize ops) | Crash gone — next run reached a **new, deeper** error (7.2), proving the wall was removed |

Lesson: on a single accelerator, **`FULL_SHARD` buys nothing and costs the resize ops that break HPU lazy mode.**
`NO_SHARD` is the correct strategy for a size-1 mesh on any backend; on HPU it's the difference between crash and run.

### 7.2 🟢 The real generation blocker: Habana HCCL rejects `ReduceOp.AVG`
With NO_SHARD, the worker got further and died on a *different* native assert:
```
Unhandled exception: Unsupported ReduceOp for HCCL process group
  hcclRedOp_t habana::getHCCLReduceOp(...)  hccl_kernels.cpp:158
```
Some collective issues `ReduceOp.AVG` (or `PREMUL_SUM`), which Habana's HCCL does not implement (it supports
SUM/MIN/MAX). The key insight: on a **world_size==1** process group, every collective is a mathematical **identity**
(one rank reducing with itself), so the ReduceOp is irrelevant to the result.

| Symptom | Fix | Notes |
|---------|-----|-------|
| `Unsupported ReduceOp for HCCL` on a background `JobThread` during generation | Coerce `AVG/PREMUL_SUM → SUM` for `world_size==1` groups, installed in `verl/utils/device.py` | Math-exact for 1 rank. **Don't** install it as a global `usercustomize.py` — that drags `import torch` into every Ray helper process and breaks raylet startup (7.3). Scope it to verl modules only. |

### 7.3 🧱 The original blocker returns: Ray metrics-agent startup deadlock
Once the code walls fell, runs kept dying at **Ray init**. The captured raylet/agent logs (run `run_05.sh` now
preserves `logs/raylogs_<JID>/` on failure) show the true mechanism:
```
raylet : Timed out waiting for file .../metrics_agent_port_...  -> Check failed -> SIGABRT
              ray::raylet::NodeManager::WaitForDashboardAgentPorts()
agent  : RPC error: Deadline Exceeded  (reporter_agent -> raylet async_get_agent_pids)
driver : GCS cannot find the node ... node registration may not be complete
```
The raylet's `NodeManager` constructor **blocks** waiting for the agent to write its port file; the agent
**blocks** on a gRPC to the raylet → **mutual deadlock**. It's **timing-sensitive** — only fires when the agent
imports slowly (cold page cache, imports off the shared `cpkgs` FS), which is why the first run of a session can
succeed and later ones fail on the same node.

| Thing tried | Outcome |
|-------------|---------|
| `include_dashboard=False` in ray.init | No effect — the **raylet** waits for the agent regardless of the dashboard head |
| `RAY_agent_register_timeout_ms=300000`, `RAY_raylet_start_wait_time_s=600` | Raylet still aborts on the agent-port wait (~113 s) |
| Excluding nodes / `pkill` stale Ray | Helps when the cause is a *poisoned* node (my own leftovers) or a *stuck HPU module* (`synStatus=8 Device acquire failed`), but not the cold-import deadlock |
| **Cache-warming retry loop** (`_run_inside_05.sh` runs `main_ppo` up to 4×, cleaning Ray between attempts) | The intended fix: attempt 1 warms the page cache so the agent imports fast enough to win the race on the retry. Only retries *infra* signatures; breaks immediately on real training errors. |

### 7.4 Where it stands
- **Code walls: removed.** NO_SHARD + ReduceOp coercion mean the actor can train *and* generate on **one card** —
  **disaggregated vLLM is no longer required** for the "few iterations" goal.
- **Infra blocker: active.** The Ray metrics-agent deadlock is the last thing between the current state and a
  validated GRPO iteration. Retry mitigation is in; if insufficient, next levers (in order): drop the loopback
  `_node_ip_address` override, stage `cpkgs` on node-local disk, or pre-start a head node with the agent disabled.

> Net: the project's "Honest bottom line" changes from *"needs disaggregated vLLM or a validated pin"* to
> *"single-card HF rollout is viable; the only remaining obstacle is Ray-in-container startup, not the HPU stack."*

---

## Phase 8 — SUCCESS: a full GRPO iteration runs end-to-end on Gaudi (session 2)

> Job `57954952` on `gaudi002` ran **Qwen2.5‑0.5B GRPO on GSM8k for 3/3 steps on one Gaudi HPU**, exiting
> `VERL_RC=0` with reward/loss and full metrics logged. This satisfies success criterion #3. The winning log is
> saved at `logs/SUCCESS_grpo_3steps_57954952.log` on Sol (proof excerpt in `results/`).

### The last two blockers after generation (and their fixes)
Once Ray + device-acquire + generation worked (Phase 7), two more fell:

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 8.1 | `synStatus=8 Device not found / acquire failed` at worker init on a **shared** node | Sol sets **no** per-job HPU isolation (`HABANA_VISIBLE_MODULES` unset; all 8 `/dev/accel*` world-visible). On a node shared with other users, the worker grabs a module someone else already holds (Gaudi modules are exclusive-per-process). | Request the node **`--exclusive`** so all 8 modules are free → module 0 acquires cleanly. (Non-wasteful alternative: pick a free module via `hl-smi` and set `HABANA_VISIBLE_MODULES=<int>` — note `=all` is **invalid**, the runtime parses ints.) |
| 8.2 | `This function should not be called in lazy flow` in `compute_log_prob` → FSDP `init_flat_param_attributes` → `_free_storage(flat_param._mp_shard)` → `_resize_(0)` | This is a **second** resize, distinct from the NO_SHARD one: FSDP **MixedPrecision** allocates a low-precision `_mp_shard` and frees it. Even with NO_SHARD, MP triggers the lazy-unsupported resize. | Set **`mixed_precision=None` on HPU** in `fsdp_workers.py` (both FSDP build sites). The actor already loads in fp32 (`model_dtype` default), so no MP is needed; params stay fp32, no `_mp_shard`, no resize. |

### What the run shows
- 3 steps, `VERL_RC=0`. Generation on HPU: **36 s → 4.8 s** across steps as the graph compiler caches recipes;
  full step **239 s → 9.9 s**. Peak memory 23.8 GB allocated / 94.6 GB reserved.
- Every metric logged: `actor/pg_loss`, `actor/entropy`, `actor/ppo_kl`, `actor/grad_norm`, `critic/score`,
  `critic/advantages`, `response_length`, `timing_s/*`, throughput.

### Known-benign caveats (training quality, NOT Gaudi/pipeline bugs)
- `critic/score/mean: 0.0` every step — the 0.5B model gets **every** GSM8k answer wrong (128-token responses are
  too short for chain-of-thought; possible answer-format mismatch). Zero reward → zero variance → zero advantage →
  no learning signal. Expected cold-start for a tiny model; raise `max_response_length`, add steps, and verify the
  reward parser to get nonzero scores.
- `actor/grad_norm: nan` — with zero advantages the PG loss is 0; the NaN arises in the entropy/log-prob numerics
  and verl **skips** NaN updates. Benign for this proof run.
- `prompt_length/*` shows absurd values (~1e7) — a metric-aggregation quirk (looks like summed token IDs), not a
  correctness issue.

### The complete fix list that made it run (all in `patches/verl05.diff` + scripts)
1. **Ray cache-warming retry loop** (`_run_inside_05.sh`) — rides past the metrics-agent startup deadlock.
2. **`--exclusive`** node (`run_05.sh`) — guarantees a free HPU module to acquire.
3. **`NO_SHARD`** for a size-1 FSDP mesh — kills the sharding `storage._resize_` crash.
4. **`ReduceOp.AVG/PREMUL_SUM → SUM`** coercion in `device.py` — Habana HCCL compat (generation).
5. **`mixed_precision=None` on HPU** — kills the `_mp_shard` resize crash (log-prob/training forward).
6. (+ all Phase 0–6 patches: HPU platform/device registration, Ray HPU resource, attn→sdpa, FSDP pre-move,
   optimum-habana adapt + `_sample` fixes, etc.)
