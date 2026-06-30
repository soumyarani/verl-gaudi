# memory.md — Project State, Checkpoints, and Exact Next Steps

> If you are Claude (or a human) picking this up: read this top-to-bottom, then `skills.md` for the run recipes.
> Everything is checkpointed on ASU Sol at **`/scratch/ssamine4/verl_gaudi/`** (SSH host alias `sol`, needs ASU VPN).

Last updated: 2026-06-30 (session 2 — the "fresh-eyes" debugging pass).

## TL;DR status
- ✅ **DONE (2026-06-30): verl GRPO ran 3/3 steps on ONE Gaudi HPU, `VERL_RC=0`, all metrics logged** (job 57954952, gaudi002). Success criterion #3 met. See Phase 8 in `03-debugging-journey.md`.
- ✅ verl **installed and runs on Intel Gaudi 2** (driver SynapseAI 1.24) — Ray + custom `PlatformHPU`/`device.py`
  HPU backend + **FSDP model sharding on the Gaudi card** + optimizer + reward all run.
- ✅ **The documented "lazy/eager catch-22" wall was a MISDIAGNOSIS and is now removed** (see below).
- ✅ **The real generation blocker (Habana HCCL rejects `ReduceOp.AVG`) is fixed** with a math-equivalent coercion.
- 🧱 **A full GRPO iteration is still UNPROVEN end-to-end.** The remaining blocker is *infrastructure*: a
  Ray-in-apptainer **metrics-agent startup deadlock** that kills the job before generation completes. Mitigation
  (cache-warming retries) is in place; validation pending.

## ⭐ The big reframe (session 2): you probably DON'T need disaggregated vLLM
The previous `memory.md` said the only way forward was **disaggregated vLLM** (separate HPUs for actor vs. rollout),
because the HF-rollout path hit a "catch-22" (FSDP `summon_full_params` needs eager; optimum-habana generate needs
lazy). **That framing was wrong.** Root cause analysis this session showed:

1. **The catch-22 was just pointless single-card FSDP.** With `n_gpus_per_node=1` the FSDP mesh has size 1, so
   `get_sharding_strategy` picked `FULL_SHARD` — which still frees/re-gathers param storage (`storage._resize_`)
   every forward, and *that* resize is what HPU lazy mode can't do. **Fix: use `NO_SHARD` for a size-1 mesh**
   (params stay resident, DDP-style, no resize ops). This **cleared the crash** — the run then died on a *new,
   deeper* error, proving the wall was gone.
2. **The new, real error: `Unsupported ReduceOp for HCCL process group`** (`hccl_kernels.cpp:158 getHCCLReduceOp`).
   Some collective uses `ReduceOp.AVG`/`PREMUL_SUM`, which Habana HCCL rejects. On a **world_size==1** group every
   collective is the identity, so the ReduceOp is mathematically irrelevant. **Fix: coerce `AVG/PREMUL_SUM → SUM`**
   for single-rank groups (installed in `verl/utils/device.py`, so only verl processes get it — NOT Ray helpers).

**Implication:** the actor can **train and generate on a single card** via the HF rollout. That is strictly simpler
than disaggregated vLLM and removes the need for it for the "few iterations" goal. Disaggregated vLLM remains the
right answer for *throughput/scale*, but is not required to prove a working iteration.

## 🧱 The remaining blocker: Ray metrics-agent startup deadlock (the ORIGINAL project blocker)
Once the code walls fell, the job kept dying at **Ray init**, before/around generation, with:
```
raylet  : Timed out waiting for file .../metrics_agent_port_...  -> Check failed -> SIGABRT
                ray::raylet::NodeManager::WaitForDashboardAgentPorts()
agent   : RPC error: Deadline Exceeded   (reporter_agent talking to the raylet)
driver  : GCS cannot find the node ... node registration may not be complete
```
**Mechanism:** the raylet's `NodeManager` constructor *blocks* in `WaitForDashboardAgentPorts()` waiting for the
metrics/dashboard agent to write its port file; the agent meanwhile blocks on a gRPC call to the raylet → mutual
deadlock. It is **timing-sensitive**: it only triggers when the agent imports slowly (cold page cache, imports off
the shared `cpkgs` filesystem). That's why the *first* run of a session sometimes succeeds and later ones fail.
- `include_dashboard=False` (added to `main_ppo.py` ray.init) does **NOT** help — the raylet waits for the agent
  regardless of the dashboard head.
- `RAY_agent_register_timeout_ms=300000` and `RAY_raylet_start_wait_time_s=600` are set but the raylet aborts on
  the agent-port wait at ~113 s anyway.

### Mitigations in place (session 2)
- **Cache-warming retry loop** in `scripts/_run_inside_05.sh`: runs `main_ppo` up to 4×; on a *transient infra*
  failure (GCS/agent/Device-acquire signatures) it kills stale Ray (`gcs_server`/`raylet`/`dashboard`/`plasma`),
  wipes `RAY_TMPDIR`, and retries. The first attempt warms the page cache so the agent imports fast enough to win
  the race on the retry. Non-infra failures break out immediately (no wasted ~12-min cycles).
- **Node hygiene**: failed runs can leave a node's Ray/HPU state wedged. `--exclude` known-bad nodes; the retry
  loop's `clean_ray()` also de-poisons. Some gaudi nodes have a **stuck HPU module** (`synStatus=8 Device acquire
  failed / Device not found`) — those need `--exclude`, not retry.

### If retries still don't crack it — next things to try (in order)
1. **Drop `_node_ip_address="127.0.0.1"`** in `main_ppo.py` ray.init and let Ray autodetect the real node IP
   (`10.139.126.x`). The agent reports the node under its real IP; forcing loopback may widen the addressing
   mismatch that makes the agent↔raylet gRPC time out.
2. Put `cpkgs` (the user-site) on **node-local disk** (`/tmp` or `/dev/shm`) before launch so agent imports are
   fast and the deadlock window closes. Copy `cpkgs` → `$TMPDIR/cpkgs`, set `PYTHONUSERBASE` there.
3. Pre-launch a head node with `ray start --head --include-dashboard=false` and `ray.init(address=...)` (memory
   notes this was historically flaky, but worth a second look with the clean-ray hygiene now in place).
4. Only if all of the above fail: fall back to **disaggregated vLLM** (the old plan).

## Cluster facts (Sol)
- Host `sol` = `login.sol.rc.asu.edu`, user `ssamine4`. **Needs ASU VPN**. **`sbatch` only** (no interactive
  `srun`/`salloc`). Run on a **gaudi node** (login node can't import HPU torch — old glibc).
- `gaudi` partition: 10 nodes, 8× HL-225 (Gaudi2, 92 GB), driver **SynapseAI 1.24**.
- **SLURM fairshare:** Gaudi is billed at the *generic* `gres/gpu=3.0` weight (no `hl225`-specific weight) — the
  cheapest accelerator on the cluster (A100=25, H100=40). A 1-HPU/16-CPU/96 GB job bills `43/min`, mostly CPU+mem.
  FairShare factor was 0.75 (healthy) and decays with a 7-day half-life. Running on Gaudi barely moves it.

## Workspace layout — `/scratch/ssamine4/verl_gaudi/`
| Path | What |
|------|------|
| `gaudi_124_pt210.sif` | Habana 1.24 **PyTorch** container (torch 2.10) — the HF-rollout / FSDP path |
| `gaudi_124_vllm.sif` | Habana 1.24 **vLLM** container (vllm 0.9.1 + vllm_gaudi) — disaggregation path (unused now) |
| `verl05/` | verl **0.5.0** clone — patched; **the active path** (HF rollout). `git diff` here == `patches/verl05.diff` |
| `verl/` | verl **0.9.0.dev0** clone (vLLM path) — patched; has `platform_hpu.py` |
| `cpkgs/` | user-site for the PT container (`PYTHONUSERBASE`): verl deps, ray 2.55.1, transformers 4.49 + optimum-habana 1.18 |
| `models/Qwen2.5-0.5B-Instruct/`, `data/gsm8k/{train,test}.parquet` | model + data |
| `scripts/` | `run_05.sh` (sbatch wrapper, captures raylet logs on fail), `_run_inside_05.sh` (retry loop) |
| `logs/` | job logs (`slurm-<JID>.out`); `logs/raylogs_<JID>/` = captured raylet/gcs/agent logs on a failed run |

## Run it
```bash
ssh sol   # ASU VPN up
cd /scratch/ssamine4/verl_gaudi
sbatch scripts/run_05.sh           # HF rollout, NO_SHARD + ReduceOp coercion + Ray retry loop
# watch: tail -f slurm-<JID>.out ; success markers = "After FSDP", "Training Progress", "actor/pg_loss"
```

## The patches that matter now (session-2 additions in **bold**)
1. `device.py`/`platform_hpu.py`: register `hpu` (device, `hccl`, `HABANA_VISIBLE_MODULES`, empty_cache shim,
   optimum-habana adapt).
2. **`fsdp_workers.py` `get_sharding_strategy`: return `NO_SHARD` when mesh size==1** (kills the resize crash).
3. **`device.py`: coerce `ReduceOp.AVG/PREMUL_SUM → SUM` on world_size==1 groups** (HCCL compat).
4. **`main_ppo.py` ray.init: `include_dashboard=False`** (minor; doesn't fix the agent deadlock by itself).
5. **`_run_inside_05.sh`: cache-warming Ray retry loop**; **`run_05.sh`: `RAY_raylet_start_wait_time_s=600`,
   raylet-log capture on failure, exclude stuck-module nodes.**
6. base.py HPU resource + ray_trainer.py HPU count + main_ppo `ray.init(resources={HPU:8})`; ray_option_utils
   fractional HPU; attn `flash_attention_2→sdpa`; FSDP pre-move to HPU; HFRollout `num_return_sequences=1`,
   inputs→HPU, drop bf16 autocast, skip summon on HPU; `transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]`;
   optimum-habana `_sample` eos-device + empty-slice guards; `trainer.use_v1=False`.

## Honest bottom line
> verl runs on Gaudi through the entire training pipeline. This session **dismantled the two "walls"** that the
> previous notes treated as fundamental: the HF-rollout "lazy/eager catch-22" was just pointless single-card FSDP
> sharding (fixed with `NO_SHARD`), and the generation crash underneath it was Habana HCCL rejecting `ReduceOp.AVG`
> (fixed with a math-equivalent `AVG→SUM` coercion for single-rank groups). **As a result, disaggregated vLLM is no
> longer required** to prove a few GRPO iterations — the actor can train and generate on one card. A full iteration
> is **not yet validated end-to-end**, because the job now dies on the *original* project blocker: a Ray-in-apptainer
> metrics-agent startup deadlock (`WaitForDashboardAgentPorts` ↔ agent gRPC). A cache-warming retry loop is in place
> to ride past its timing-sensitive window; if that proves insufficient, the next levers are dropping the loopback
> `_node_ip_address` override and staging `cpkgs` on node-local disk. All patches and both environments are
> checkpointed on `/scratch/ssamine4/verl_gaudi/`.
