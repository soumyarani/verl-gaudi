# vllm-gaudi branch — enabling the vLLM rollout on Gaudi

> **Why this branch exists.** The [benchmark](BENCHMARK.md) proved the point: verl's **HF rollout** is not viable
> for real settings on Gaudi (one full-settings step's generation ran **>38 min and never finished**, because HF
> `model.generate` is serial with no paged/batched KV cache). A100 does the same step in ~110 s *only because it
> uses vLLM*. So the single highest-value fix for Gaudi is to make verl's **vLLM rollout** work on HPU.
>
> This branch tracks that work, isolated from `main` (which holds the working HF-rollout port + benchmarks).

## The core problem (from earlier attempts, documented on `main`)
verl colocates the FSDP **actor** and the **rollout** on the same accelerator ("hybrid engine"). On CUDA that's
free (one process, multiple contexts). On Gaudi, **modules are exclusive-per-process** — the vLLM rollout runs as a
*separate* process and can't acquire the module the FSDP actor already holds -> `synStatus=8 Device acquire failed`.

Two ways out, in order of effort:
1. **Standalone validation first (this step).** Confirm `vllm_gaudi` can load Qwen2.5-0.5B and generate on *one*
   HPU with no verl/Ray in the picture. If the engine itself works, the integration is "just" placement.
2. **Disaggregated placement.** Give the actor and the vLLM rollout **different** HPU modules (actor on module 0,
   vLLM on module 1) instead of colocating. Then there is no device-acquire conflict. This is the real target.

## Assets (on Sol, `/scratch/ssamine4/verl_gaudi/`)
- `gaudi_124_vllm.sif` — Habana 1.24 container with **`vllm` + `vllm_gaudi` (0.21)**.
- `cpkgs_vllm/` — writable user-site for that container (verl deps, patched Ray).
- `verl/` — verl **0.9.0.dev0** clone with `platform_hpu.py` + the vLLM-path patches.
- `vllm-gaudi-scripts/` (this repo) — `vllm_gaudi_smoke.{py,sh}` (standalone test), `run_vllm.sh` /
  `_run_inside_vllm.sh` (verl+vLLM path), `setup_vllm.sh`.

## Container gotchas already found (standalone step)
- vLLM writes its compile cache to `~/.cache/vllm`; under `--no-home` that is read-only -> set
  `XDG_CACHE_HOME` / `VLLM_CACHE_ROOT` / `HOME` to a writable scratch dir.
- The V1 engine spawns an `EngineCore` subprocess that re-imports the entry script -> guard the body with
  `if __name__ == "__main__":`, and set `VLLM_ENABLE_V1_MULTIPROCESSING=0` for a single-HPU test.

## Plan / status
- [x] **Step 1 — standalone vLLM-Gaudi generation** ✅ **DONE.** `vllm_gaudi` (0.21) loaded on 1 HPU, Qwen2.5-0.5B
      init 27.8 s, **generate 2.84 s / 2 prompts, correct output** ("2+2 is equal to 4", "capital of France is Paris").
      Proves the engine works standalone -> the earlier failure was *only* colocation, not the engine. See
      `vllm-gaudi-scripts/vllm_gaudi_smoke_OK.txt`.
- [~] **Step 2 — vLLM rollout in verl (verl 0.9), single card.** **Surprise: the device-acquire wall is GONE.**
      verl 0.9's rollout is a `vLLMHttpServer` (separate Ray actor) that gets its own HPU, so it reached
      **weight sync** (`After FSDP, 3.69 GB on HPU` -> `actor_rollout_update_weights`). New blocker there:
      `This function should not be called in lazy flow` — the FSDP gather during weight sync, the same
      resize issue NO_SHARD fixes. Applied **NO_SHARD + mixed_precision=None to verl 0.9** (`patches/verl09_main.diff`,
      `engine/fsdp/utils.py` + `transformer_impl.py`) and re-testing.
- [~] **Step 3 — disaggregated placement (`separate_async`)** — MAJOR PROGRESS: hang fixed (num_cpus), reaches actor FSDP; blocked at the weight-sync communicator (Habana stateless HCCL missing). See below.
      Applied: `replica.py` `device_name=get_device_name()` (HPU fix); config
      `trainer.use_v1=True trainer.v1.trainer_mode=separate_async`, `rollout.mode=async`,
      `rollout.nnodes=1 rollout.n_gpus_per_node=1`, `rollout.checkpoint_engine.backend=hccl`,
      `ppo_mini_batch_size==train_batch_size`; installed `TransferQueue==0.1.8` (v1 dep, PyPI; note it bumps
      tensordict to 0.13 vs verl's <=0.10 — watch for conflicts). Result: **`[validate_config] passed` and
      `TaskRunnerV1` starts** — the disaggregated trainer is live — but v1 setup (transfer_queue + standalone
      rollout pool + vLLM on the 2nd HPU + hccl weight sync) did not finish within 55 min; relaunched with a 2.5 h
      limit to see slow-vs-stuck. Scripts: `vllm-gaudi-scripts/run_vllm_disagg.sh`, `_run_inside_vllm_disagg.sh`.
      Remaining unknowns on HPU: transfer_queue init, the standalone vLLM pool acquiring a 2nd module, and the
      hccl weight transfer between the two pools.
- [ ] **Step 4 — full GRPO iteration with vLLM rollout on Gaudi**, then benchmark vs HF-rollout and A100.

## verl 0.9 supports disaggregated rollout natively (research finding)

`trainer.use_v1=True` + `trainer.v1.trainer_mode=separate_async` builds a **standalone vLLM server in its own Ray
resource pool on separate devices** from the actor (`workers/rollout/replica.py:189 init_standalone`), syncing
weights over `rollout.checkpoint_engine.backend` (**`hccl`** is supported = Gaudi). Config: `rollout.mode=async`,
`rollout.nnodes=1`, `rollout.n_gpus_per_node=<sep pool>`, `train_batch_size==ppo_mini_batch_size`. **One HPU code
gap:** `replica.py:184,223` hardcodes `device_name="cuda"/"npu"` — needs an `hpu` branch so the standalone worker
group passes the platform check in `single_controller/ray/base.py`. This is Path B (true disaggregation) if the
colocated NO_SHARD fix (Step 2) isn't enough.

## Reproduce step 1
```bash
sbatch vllm-gaudi-scripts/vllm_gaudi_smoke.sh   # exclusive Gaudi node; prints VLLM_GAUDI_SMOKE_OK on success
```


### Step 3 — current blocker (as of last run): v1 trainer hangs at init on HPU
Two long runs (55 min, then 2.5 h) both hit the SLURM time limit with the **same signature**: after
`TaskRunnerV1 ... Platform override from VERL_PLATFORM: hpu` and a `verl/workers/engine/mindspeed/transformer_impl.py`
NPU-router-replay warning, there is **zero further output** until the job is killed. So the v1 disaggregated trainer
**deadlocks at init**, before creating any resource pool, the standalone vLLM server, or FSDP.

Suspects (v1-init, in order): (1) **TransferQueue** controller/server startup (Ray-based) deadlocking on HPU — this
is the v1 data plane and is brand-new here; (2) the v1 trainer importing the **mindspeed/megatron NPU** path and
stalling; (3) another Ray-in-container startup deadlock like the metrics-agent one on `main` (but `Started a local
Ray instance` did print, so ray.init itself completed).

**Concrete next diagnostic (do this before any more long runs):** resubmit with a short (~15 min) limit and, from
the batch script, after ~4 min `py-spy dump --pid <TaskRunnerV1 pid>` (install `py-spy` in `cpkgs_vllm`) to capture
the exact Python frame it is stuck on. That single stack trace tells you whether it's TransferQueue, mindspeed, or
Ray — and turns this from a blind multi-hour loop into a targeted fix. Also resolve the `tensordict` tension
(TransferQueue 0.1.8 pulled 0.13; verl wants <=0.10) since a silent version mismatch could itself hang.


### Step 3 — ROOT CAUSE FOUND (py-spy) + fix
A py-spy dump of the hung `ray::TaskRunnerV1` gave the exact frame:
```
ray.get()  <- blocks forever
get_placement_group        (transfer_queue/utils/common.py:43)
initialize_simple_storage  (transfer_queue/storage/bootstrap/simple_storage_bootstrap.py:37)
_maybe_create_tq_storage   (transfer_queue/interface.py:75)
init                       (transfer_queue/interface.py:191)
run                        (verl/trainer/main_ppo.py:143)
```
TransferQueue's `SimpleStorage` init builds a Ray **placement group** and `ray.get(pg.ready())` never
returns. The `TransferQueueController` actor itself is healthy (idle on a zmq poll).

**Why it hangs (verified):** verl overrides `transfer_queue...SimpleStorage.num_data_storage_units` to **8**
(`ppo_trainer.yaml:391`, vs the TransferQueue package default of 2). So the storage PG requests **8 × {CPU:1}**.
Our launch capped Ray at `ray_init.num_cpus=8`, and at `tq.init` (which runs *before* any FSDP/vLLM pool) the
`TaskRunnerV1` (1 CPU) + `TransferQueueController` (1 CPU) already hold 2 → only **6 free < 8 requested** → the PG
pends forever → `pg.ready()` blocks.

**Fix (config-only, chosen via a 5-way workflow, confidence 0.8):** raise the Ray CPU cap so the storage PG (and
the later FSDP/vLLM pools drawn from the same cap) fit. `_run_inside_vllm_disagg.sh`:
`ray_kwargs.ray_init.num_cpus=8 → 64`, and bump SLURM `-c 16 → 96` so the cgroup allows it (node is `--exclusive`,
152 cores). Fallbacks if it still hangs: shrink the PG via `+...SimpleStorage.num_data_storage_units=1`; then check
for a leaked/stale Ray session (kill gcs_server/raylet, wipe the ray tmp) before re-running.


### Step 3 — num_cpus fix CONFIRMED WORKING; next blocker = disaggregated weight-sync communicator
The `num_cpus=64` fix cleared the `tq.init` hang **completely** — the run advanced through TransferQueue init,
the resource pools, and built the **actor FSDP on HPU** (`After FSDP, memory allocated 3.69 GB`). Diagnosis
confirmed. It then died on a clean new error: `ValueError: Checkpoint engine hccl not registered`.

**Next blocker (characterised, not yet fixed):** the disaggregated weight transfer (train actor HPU → vLLM rollout
HPU) has no Habana implementation:
- verl's HCCL checkpoint engine (`verl/checkpoint_engine/hccl_checkpoint_engine.py`) registers under the name
  **`nccl`** but is **gated to Ascend NPU**: `if not is_torch_npu_available(): raise ImportError` (line 30). On
  Gaudi `is_torch_npu_available()` is False, so it never loads → `hccl`/`nccl` resolve to the CUDA NCCL engine.
- Even if the guard is relaxed and its `torch.npu.*` calls are routed through verl's device facade
  (`get_torch_device()`), the deeper gap is the **stateless communicator**: `stateless_init_process_group`
  (`verl/utils/distributed.py:99`) imports `vllm_ascend`'s `PyHcclCommunicator` on NPU, else the CUDA
  `PyNcclCommunicator`. **`vllm_gaudi` provides only an in-engine `HpuCommunicator` (DeviceCommunicatorBase), not a
  stateless PyHccl** for cross-job weight sync. So a Habana stateless-HCCL communicator must be written (or the
  engine re-implemented on plain `torch.distributed` with the `hccl` backend + a TCPStore rendezvous).

This is a real implementation task, not a config fix — it's the last mile of disaggregated vLLM on Gaudi and a
natural upstream contribution (making verl's HCCL checkpoint engine + stateless communicator work on Habana HPU,
not just Ascend NPU). Concrete sub-steps: (1) relax the HCCL-engine import guard to `npu OR hpu`; (2) route its
`torch.npu.*` device calls through `get_torch_device()`/`get_device_name()`; (3) provide a Habana stateless HCCL
communicator for `stateless_init_process_group` (wrap habana hccl like `vllm_ascend`'s PyHccl, or use
`torch.distributed` + `hccl` over a TCPStore).


### Step 3 — weight-sync plugin WORKS through bucketing; stall in the cross-process transfer
The `custom_backend_module` plugin (`patches/plugin/checkpoint_engine/hccl_hpu.py`) cleared every checkpoint-engine
error in sequence, each fixed:
1. `hccl not registered` → plugin registers the device-ported HCCL engine as `hccl` (verified in-container).
2. `StatelessProcessGroup.__init__() got 'socket'` → use `StatelessProcessGroup.create(host,port,rank,world_size)`.
3. `Weight embed_tokens ... too large to fit in the bucket` (fp32 ~520 MiB) → `update_weights_bucket_megabytes=768`.

The run now reaches the **actual weight transfer**: actor gathers weights (`NO_SHARD full_state_dict`), the vLLM
rollout server loads its base checkpoint on the 2nd HPU — then **stalls in the weight-sync region** (neither side
logs `init_process_group rank`), hitting the 70-min limit. The suspected cause is the rendezvous/collective barrier
(`all_gather_obj`) where actor rank 0 and the vLLM rollout rank 1 must both join — if the v1 orchestration doesn't
trigger the rollout's `receive_weights` in lockstep with the actor's `send_weights`, one side blocks. A py-spy
diagnostic of both processes at the stall is running to pinpoint it (same tool that nailed the tq hang).

NET: the disaggregated path is now working end-to-end up to and INCLUDING the weight-transfer machinery
(registration + communicator + bucketing all functional). The last item is the transfer rendezvous/orchestration —
a specific, diagnosable stall, not a missing component.


### Step 3 — py-spy ROOT CAUSE of the transfer stall: it's a 4-layer stack; layer 4 is vLLM-Gaudi internals
A py-spy dump of every process at the stall (saved: `vllm-gaudi-scripts/pyspy_weightsync_stall.txt`) gives the
definitive picture. Disaggregated vLLM weight-sync on Gaudi is a **4-layer stack**:

1. **TransferQueue storage placement group** — FIXED (`ray_init.num_cpus=64`).
2. **checkpoint engine, actor→rollout process** (verl's HCCL engine) — FIXED by our plugin
   (`patches/plugin/checkpoint_engine/hccl_hpu.py`): registration, StatelessProcessGroup.create(), 768 MiB bucket.
   **This layer WORKS** — the rollout side received a weight bucket (its `BucketedWeightSender.async_send_weights`
   reached `self.socket.recv()` at `bucketed_weight_transfer.py:132`, which only happens after a bucket arrives).
3. **bucketing / metadata over ZMQ** — works.
4. **rollout process → vLLM engine, via `update_weights_from_ipc` + shared memory** — **THE BLOCKER.** On Gaudi
   `use_shm = not is_support_ipc()` is True (the intended non-CUDA path), so the sender writes buckets to shared
   memory, sends `bucket_meta` over ZMQ, and blocks on `socket.recv()` waiting for the vLLM engine worker to consume
   the bucket and ACK. But the py-spy shows the vLLM **`EngineCore` sitting idle in its normal
   `_process_input_queue` zmq poll** (`vllm/v1/engine/core.py:1194`) — it never executes `update_weights_from_ipc`,
   so it never reads the shm bucket or ACKs → the sender hangs forever.

**What this means:** the actor→rollout weight *delivery* on Gaudi is solved (our plugin). The remaining gap is the
rollout→engine *load* path — `update_weights_from_ipc` is not wired/executed on the vLLM-Gaudi **v1 async engine**.
That is **vllm_gaudi / vLLM-v1-engine internals**, a substantial separate component (the async server's handling of
the weight-update RPC + shared-memory read on HPU), not a verl-side config or a one-file port.

**Where this leaves the branch (honest):** disaggregated vLLM rollout on Gaudi is working through **3 of its 4
layers** — a genuinely novel result (it had never run at all before). The 4th layer needs work inside vllm_gaudi's
weight-update path. Candidate next directions: (a) check whether the vLLM-Gaudi **async** engine supports the
`update_weights_from_ipc` collective RPC at all (it may only support sync/colocated weight updates) and, if not, try
`trainer_mode=colocate_async` instead of `separate_async`; (b) file/adapt against vllm_gaudi for HPU shared-memory
weight loading; (c) implement the shm read in a vllm_gaudi worker method. This is upstream-contribution territory,
not a quick fix.


### Step 3 — fallback confirmed: colocate_async hits the SAME layer-4 wall
Tried `trainer.v1.trainer_mode=colocate_async` (the workflow's fallback). It reached the identical weight-sync
region (vLLM server up, actor `full_state_dict`) and stalled to the time limit — same as `separate_async`. Confirmed:
both v1 async modes share the `update_weights_from_ipc` + shared-memory path into the vLLM engine, so a trainer-mode
config change cannot route around the missing engine-side method. The blocker is squarely in vllm_gaudi / the
vLLM-v1 async engine's weight-update path.

**Final status of the vllm-gaudi branch:** every tractable verl-side / plugin-side fix is applied and works;
disaggregated vLLM rollout on Gaudi runs through the entire weight-*delivery* path (TransferQueue, the HCCL
checkpoint-engine plugin actor→rollout, bucketing). The one remaining gap is the weight-*loading* path
(`update_weights_from_ipc` reading HPU shared-memory buckets inside the vLLM engine) — a vllm_gaudi contribution,
not a config/one-file port.