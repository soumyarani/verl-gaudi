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
