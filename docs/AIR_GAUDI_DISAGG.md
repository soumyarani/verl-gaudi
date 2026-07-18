# Disaggregated GRPO on Intel Gaudi 2 (ASU AIR / Kubernetes) — the working pipeline

**Status: ✅ WORKING.** A full **disaggregated** GRPO iteration runs end-to-end on Gaudi 2 HPUs:
`generation → reward → advantage → update_actor`, for 3/3 steps, clean exit (`VERL_RC=0`), with
`actor/pg_loss` and the full timing/metrics block logged.

```
step:1 ... actor/pg_loss:0.0 ... response_length/mean:125.16 ... num_turns:2 ...
           timing_s/gen:21.8 - timing_s/old_log_prob:0.05 - timing_s/adv:0.06 - timing_s/update_actor:457.6
step:2 ... step:3 ...
VERL_RC=0
```

- **Stack:** verl `0.9.0.dev` (@1ff76cc) + `vllm_gaudi 0.24` (which pulls vLLM `0.23.1rc1`) on SynapseAI `1.24`.
- **Config:** `trainer.use_v1=True trainer.v1.trainer_mode=separate_async` — the **disaggregated** path where the
  actor (FSDP trainer) and the vLLM rollout run in **separate processes on separate HPUs**, and trained weights are
  streamed actor→rollout each step through a checkpoint engine.
- **Where:** ASU **AIR Platform** — a dedicated Gaudi-2 **Kubernetes** cluster (Rancher), namespace `user-ssamine4`,
  quota **max 4 HPUs**.

> This is the disaggregated counterpart to the **colocated** single-card GRPO documented in
> [`03-debugging-journey.md`](03-debugging-journey.md) (Sol, HF-rollout, `NO_SHARD` + `ReduceOp AVG→SUM`). That one
> proved the *training* half on HPU; this one adds the **vLLM rollout + disaggregated weight-sync** half, which the
> colocated path never exercised.

---

## 1. Why AIR / Kubernetes (the pivot)

On Sol's shared `gaudi` partition the disaggregated weight-sync stalled at **layer 4**: the vLLM EngineCore on
`vllm_gaudi 0.21` sat idle and never executed `update_weights_from_ipc`, so the actor's weight push hung forever
(see [`VLLM_GAUDI.md`] / the `vllm-gaudi` branch). The fix needed a **current, bring-your-own `vllm_gaudi` stack**,
which containers give cleanly. ASU's **AIR Platform** provides raw Gaudi-2 K8s compute (not just the CreateAI
model-API), so the work moved there.

On `vllm_gaudi 0.24` the EngineCore **does** execute `update_weights_from_ipc` and runs `receive_weights` — the
architectural gap is closed. Everything below is what it then took to get from "the transfer starts" to
"3 training steps complete."

---

## 2. The disaggregated pipeline (what actually runs)

```
                         Ray single-controller (TaskRunnerV1)
                                        │
        ┌───────────────────────────────┼────────────────────────────────┐
        │                               │                                 │
  ┌─────▼──────┐                 ┌───────▼────────┐               ┌────────▼─────────┐
  │  Actor      │  weights (ZMQ  │ CheckpointEngine│  shm buckets  │ vLLM rollout      │
  │  (FSDP,     │  bucketed shm) │  (CPU-only,     │──────────────▶│  replicas ×2      │
  │   1 HPU)    │───────────────▶│   hccl_hpu)     │               │  (HpuModelAdapter,│
  │             │                └────────────────┘               │   1 HPU each)     │
  │  update_    │                                                 │  generate()       │
  │  actor      │◀──── rollout trajectories (TransferQueue) ───────│                   │
  └─────────────┘                                                 └───────────────────┘
        │                                                                   │
        └───────── old_log_prob (actor recompute) · advantage (GRPO) ───────┘

  Per step:  gen → reward → advantage → compute_old_log_prob → update_actor → sync weights → repeat
```

- **`separate_async`** puts the actor and the rollout in different processes. 4 HPUs are consumed: **1 actor + 2 vLLM
  rollout replicas** (`rollout.n=4` samples, 2 replicas) + the checkpoint-engine worker (made **CPU-only**, so it needs
  none). With quota=4 this just fits.
- **TransferQueue** (`transfer_queue`, capital-T package) carries prompts/trajectories between controller and workers.
- **Checkpoint engine** (`hccl_hpu` custom backend) streams the trained actor weights to the rollout replicas each
  step as **bucketed shared-memory** transfers with a per-bucket ZMQ ACK.

---

## 3. Environment (the K8s substrate)

| Piece | Value / requirement |
|---|---|
| Cluster | AIR Platform, Rancher `rancher.rc.asu.edu`, namespace `user-ssamine4` (namespaced SA) |
| Nodes | Gaudi2 HL-225 `dcx-gaudi0xx`, driver **1.24.0**, **quota = 4 HPU** (`gpu-quota`) |
| Image | `vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0` (torch 2.11 ↔ vllm_gaudi 0.24) |
| **★ Device access** | **`runtimeClassName: habana` on the pod spec** — auto-injects `/dev/infiniband` uverbs char devices. Without it HCL's `g_ibv.init` fails → `synStatus=26 Device acquire failed`, even for `torch.ones(3, device='hpu')`. Kubewarden blocks hostNetwork/hostPath/SYS_NICE/IPC_LOCK, so runtimeClass is the **only** route. |
| IPC | `hostIPC: true` (Habana shm) allowed |
| Persistence | runs in a **tmux session inside the pod** (survives laptop offline). `/workspace` is `emptyDir` (lost on pod restart — see follow-ups). |

Full K8s access + install details: [`AIR_RUNBOOK.md`] and the `air-gaudi-k8s` project memory.

---

## 4. The full fix chain (layer by layer)

Each fix unblocked the next. Grouped by the layer of the pipeline it lives in.

### Layer A — Infra / Ray / K8s
| # | Change | Why |
|---|---|---|
| A1 | `runtimeClassName: habana` on the pod | inject `/dev/infiniband` → HPU device acquire works (see §3) |
| A2 | `single_controller/ray/base.py` ResourcePoolManager `max_colocate_count 3→1`; `workers/rollout/replica.py` `init_standalone 2→1` | Ray natively **rejects fractional custom resources** (HPU), unlike GPU. Whole-device per worker. |
| A3 | Run verl from `/root` (not `/workspace`) | the cloned `vllm/` source dir under `/workspace` shadows the installed `vllm` package → `cannot import name 'LLM'` |
| A4 | 1.24.1 image (torch 2.11) | matches vllm_gaudi 0.24; avoids `torchvision::nms does not exist` |

### Layer B — Checkpoint engine (rollout-side weight receiver)
| # | Change | Why |
|---|---|---|
| B1 | **CPU-only** checkpoint engine (`hccl_hpu.py`: receive-side device/buffers = `cpu`, keyed on `is_master`) | it only stages weights through shm; making it CPU-only means the `CheckpointEngineWorker` needs **no HPU** → frees a device (`synStatus=8 Device not found` otherwise) |
| B2 | `hccl_hpu._hpu_stateless_init_process_group`: use `StatelessProcessGroup.create()` not the raw ctor | the raw ctor signature changed (`socket` kwarg) |
| B3 | `bucketed_weight_transfer.py::_cleanup`: guard `get_torch_device().ipc_collect()` with `hasattr` | `habana_frameworks.torch.hpu` has no `ipc_collect` (a CUDA-ism) |
| B4 | `vllm_rollout.py` device_uuid guarded by `is_support_ipc()` | HPU doesn't support IPC handles |

### Layer C — Weight sync into the vLLM model  ← **THE "deadlock"**
**Symptom:** the actor's `async_send_weights` blocks forever on `socket.recv()` (`bucketed_weight_transfer.py:132`) —
looks like a hang. **It is not a deadlock.** The vLLM worker's `receive_weights → load_weights` **raises**, so the
per-bucket ZMQ **ACK is skipped**, and the sender waits on an ACK that never comes.

**Root cause:** HPU's `model.to("hpu")` **downgrades vLLM's stateful weight-parameter subclasses**
(`ModelWeightParameter`, `PackedvLLMParameter`, …) back to a plain `torch.nn.Parameter`. That strips both:
- the **class methods** the v2 loader dispatches to (`load_qkv_weight`, `load_merged_column_weight`,
  `load_row_parallel_weight`), and
- the **instance state** those methods read (`tp_rank`, `tp_size`, `output_dim`, `input_dim`, `weight_loader`,
  `shard_id`, …).

So `model.load_weights()` during the RL weight-sync fails, and the failure **walks one attribute at a time**:
`'Parameter' has no attribute 'weight_loader'` → shape `AssertionError` (missing `shard_id`/`output_dim`) →
`'Parameter' has no attribute 'load_qkv_weight'` → (after restoring the class) `'ModelWeightParameter' has no
attribute 'tp_rank'`. Restoring a *fixed list* of attributes never converges — each restored one reveals the next.

**Fix — restore the whole subclass, class + full `__dict__`:**

1. **Capture** immediately after `get_model()`, before `.to("hpu")` has run its course
   (`vllm_gaudi/v1/worker/hpu_model_runner.py`):
   ```python
   self.model = get_model(vllm_config=self.vllm_config)
   try:
       self._verl_saved_meta = {}
       for _vn, _vp in self.model.named_parameters():
           _vm = dict(getattr(_vp, '__dict__', {}) or {})   # ALL instance state
           _vm['_cls'] = type(_vp)                           # the subclass itself
           self._verl_saved_meta[_vn] = _vm
   except Exception:
       self._verl_saved_meta = {}
   ```

2. **Restore** just before `model.load_weights()` in the RL weight-sync
   (`verl/workers/rollout/vllm_rollout/utils.py::_update_weights`):
   ```python
   _meta = getattr(getattr(self, 'model_runner', None), '_verl_saved_meta', None) or {}
   for model in self._iter_all_models():
       if _meta:
           for _n, _p in model.named_parameters():
               _sv = _meta.get(_n) or _suffix_match(_n, _meta)  # capture/restore names differ by a prefix
               if _sv:
                   _cls = _sv.get('_cls')
                   if _cls is not None and type(_p) is not _cls:
                       try: _p.__class__ = _cls                  # bring back the methods
                       except Exception: pass
                   try: _p.__dict__.update({k: v for k, v in _sv.items() if k != '_cls'})  # and the state
                   except Exception: pass
       model.load_weights(param_updates)
   ```
   Capturing the **entire `__dict__`** (not a hand-picked list) is the load-bearing detail. Names differ between
   capture (`model.layers.…`) and restore (`model.model.layers.…`), so match by suffix.

3. Plus a one-line **delegation** so the graph-wrapped adapter forwards the call
   (`hpu_model_runner.py`, `HpuModelAdapter`):
   ```python
   def load_weights(self, *args, **kwargs):
       return self.model.load_weights(*args, **kwargs)
   ```

→ **0 `LOAD_WEIGHTS_FAILED`; weight sync completes.** *(This is why the earlier "verl0.9 ↔ vLLM-0.23 version
mismatch, needs a verl-main re-port" reading was a red herring — the mismatch is real but the actual blocker was
subclass state loss, fixable in place.)*

### Layer D — Rollout generation (HPU garbage logprobs)
**Symptom:** generation crashes with `OverflowError: out of range integral type conversion attempted` inside vLLM's
async output_handler, and `KeyError: <token_id>` at `vllm_async_server.py:586`.

**Root cause:** vllm_gaudi's HPU **top-k / logprobs op emits garbage token-ids** (e.g. `-2117107376`, near INT32_MIN)
in the logprob buffer. Two consequences:
- vLLM's output_handler detokenizes those ids → the Rust `tokenizers` binding overflows → **EngineCore dies** →
  `EngineGenerateError` for all requests → empty batch → `AssertionError: number of items:[0] < k_partitions:[1]`.
- the sampled token is sometimes **absent** from the HPU logprobs dict → `logprobs[token_ids[i]]` `KeyError`.

The **logprob float values are fine**; only the token-id indices used for *cosmetic detokenization* are garbage, and
verl consumes only the numeric logprobs. Two small fixes:

| # | Change | File |
|---|---|---|
| D1 | **neuter cosmetic logprob detokenization** — return `""` per token without calling `tokenizer.decode` (response text uses a *separate* incremental detokenizer, unaffected) | `vllm/tokenizers/detokenizer_utils.py::convert_ids_list_to_tokens` |
| D2 | tolerate a missing sampled-token logprob (default `0.0`) | `verl/workers/rollout/vllm_rollout/vllm_async_server.py:586` |

```python
# D1
for token_id in token_ids:
    token_str_lst.append("")          # skip decode entirely (HPU ids can be out-of-range)

# D2
log_probs = [(logprobs[token_ids[i]].logprob
              if (logprobs is not None and token_ids[i] in logprobs) else 0.0)
             for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
```

### Layer E — `old_log_prob` plumbing (keep it default)
Do **not** set `rollout.calculate_log_probs=False`. Disabling it removes the `rollout_log_probs` key that
`trainer_base.py::_compute_old_log_prob` pops (→ `KeyError: 'rollout_log_probs'`, empty `Keys=[]`). Keep it at the
default **`True`** so the key exists; with D1 the logprobs path no longer crashes, and the numeric `rollout_log_probs`
flow through. (The `old_log_probs` used for the GRPO ratio are still recomputed correctly by the actor.)

---

## 5. Reproduce

```bash
# on your laptop: VPN → cluster
export KUBECONFIG=~/.kube/config
POD=$(kubectl get pods -n user-ssamine4 -l app=verl-dev -o jsonpath='{.items[0].metadata.name}')

# inside the pod, in a persistent tmux session (survives disconnect):
kubectl exec -it -n user-ssamine4 $POD -- tmux new -As verl
#   > bash /workspace/run_disagg_hpu4.sh 2>&1 | tee /workspace/disagg4.log
```

Watch `/workspace/disagg4.log` for `step:1 … actor/pg_loss …` → `VERL_RC=0`. On a `load_weights` failure, the real
reason is logged as `LOAD_WEIGHTS_FAILED n=… err=…`. The run script is [`scripts/run_disagg_hpu4.sh`](../scripts/run_disagg_hpu4.sh)
(the key knobs: `trainer.v1.trainer_mode=separate_async`, `rollout.n=4`, `+ray_kwargs.ray_init.resources={HPU:4}`,
`checkpoint_engine.custom_backend_module=verl.plugin.checkpoint_engine.hccl_hpu`, `trainer.device=hpu`).

### Applying the patches
The AIR-specific source edits are idempotent scripts in [`patches/air/`](../patches/air/) (run inside the pod against
the live installs):
```bash
python patches/air/patch_air_capture_restore_base.py   # add capture + restore-then-load scaffold
python patches/air/patch_air_weightsync_dict.py        # make capture = full __dict__ + __class__   (Layer C)
python patches/air/patch_air_detok_neuter.py           # skip cosmetic logprob detok                (Layer D1)
python patches/air/patch_air_logprob_tolerant.py       # default 0.0 for missing sampled logprob    (Layer D2)
```
The as-built full files are archived under [`patches/air/fixes_final/`](../patches/air/fixes_final/) for reference
(`hpu_model_runner.py`, `vllm_rollout_utils.py`, `vllm_async_server.py`, `detokenizer_utils.py`). The Layer-A/B
changes (Ray colocate count, `hccl_hpu` CPU-only checkpoint engine, `StatelessProcessGroup.create`, `ipc_collect`
guard) are documented in [`AIR_RUNBOOK.md`] / the `hccl_hpu` plugin on the `vllm-gaudi` branch.

---

## 6. Going from "it runs" to "it trains" — GSM8k GRPO run

The smoke run above proves the *machinery*, but its reward was `0.0`. Getting a real **learning signal** needed one
fix; getting a **sustained multi-step** run hits genuine HPU-runtime walls.

### 6.1 Reward = 0 was a format mismatch (fixed)
Qwen2.5-0.5B-Instruct **solves** GSM8k but answers in its native `\boxed{72}` format, while verl's `gsm8k` reward
defaults to `method="strict"` (requires the literal `#### 72`). So every sample scored 0 → advantage 0 → `pg_loss`
exactly `0.0` (no learning) — even though the model was correct. **It is not truncation** (raising
`max_response_length` 128→256 did not help). Fix: [`patches/air/patch_air_gsm8k_flexible.py`](../patches/air/patch_air_gsm8k_flexible.py)
defaults the scorer to `flexible` (last-number extraction): `\boxed{72}`→1.0, wrong→0.0.

**Result — a real GRPO step on Gaudi** (`scripts/run_grpo_resilient_v05.sh`, response 256, flexible reward):
```
step:1  critic/rewards/mean:0.1406  critic/rewards/max:1.0  actor/pg_loss:0.00747  actor/grad_norm:1.008
        response_length/mean:242.7  (pg_loss was exactly 0.0 before the reward fix)
```
Non-zero reward *and* non-zero policy-gradient loss = genuine learning signal. Logged to wandb (offline).

### 6.2 Why a *full epoch* isn't practical on this stack (yet)
Three HPU-runtime issues, diagnosed but not fixed (they are vllm_gaudi/habana-internal):
1. **`update_actor` ≈ 900-1000s/step** (response 256) — HPU graph recompilation on dynamic shapes (constant
   "was not warmed-up"). A full GSM8k epoch (~467 steps) ≈ 5 days.
2. **Intermittent actor segfault in the backward** — `habana_lazy::HbLazyTensorImpl::handle_view_cycles` ←
   `VariableHooks::set_data` (a `param.data=` swap) in a backward pre-hook, ~every 1-3 steps. Not memory (90GB HPU /
   280GB host free), not gradient checkpointing (removing it didn't help), not deterministic (one run did 2 steps
   then died on step 3).
3. **Checkpoint LOAD (resume) fails with `synStatus 26` "Graph compile failed"** (`fsdp_checkpoint_manager.load_checkpoint`)
   — the project's original graph-compile blocker, on the resume path. So **save works but resume crashes**, which
   defeats a retry-on-crash wrapper (`run_grpo_resilient_v05.sh` banks step 1, then every resume crashes on load).

**Bottom line:** the pipeline and the reward are correct and a real training step runs; a *sustained* run needs the
HPU lazy-mode backward segfault (#2) and the synStatus-26-on-checkpoint-load (#3) fixed — or, untried, eager mode
(`PT_HPU_LAZY_MODE=0`) to sidestep the lazy-view bugs.

## 7. Follow-ups (not blocking "it runs")

- **Reward is currently all-0** on this smoke config: `data.max_response_length=128` truncates ~95% of GSM8k answers
  (`response_length/clip_ratio≈0.95`) before the `#### <answer>` the reward checks → 0 reward → 0 advantage → `pg_loss=0`.
  For a *training* run raise `max_response_length` to 512 and enable wandb logging. (Infra is fine; this is a config/eval matter.)
- **Persist `/workspace`** — it's `emptyDir`; a pod restart loses the checkpoints. Needs a working PVC (default Longhorn
  fails `topologyKeys not found`).
- **Upstream** the Gaudi fixes (the subclass-restore is a general HPU-`.to()` issue; the logprob detok is a vllm_gaudi bug).
