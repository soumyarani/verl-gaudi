# verl-gaudi

Running **[verl](https://github.com/volcengine/verl)** (the HybridFlow RL post-training framework) on **Intel
Gaudi 2 (HPU)** accelerators — the full port, every patch, both container environments, and a from-first-principles
explanation of *why* each change was needed.

Target run: GRPO post-training of **Qwen2.5-0.5B-Instruct** on **GSM8k**, on the ASU **Sol** supercomputer's
`gaudi` partition (8× HL-225, SynapseAI 1.24).

> **Reproducing from scratch?** Read [`SETUP.md`](SETUP.md) — exact, tested, step-by-step. The complete
> Gaudi-modified verl source is in [`verl05-gaudi/`](verl05-gaudi/) (full tree) and `patches/verl05.diff` (diff).

---

## What this repo is

This is both a **working checkpoint** and a **teaching artifact**. If you only read one thing, read
[`docs/01-verl-pipeline.md`](docs/01-verl-pipeline.md) then [`docs/02-gaudi-port.md`](docs/02-gaudi-port.md): together
they explain LLM RL post-training from scratch and then show exactly which PyTorch/backend assumptions break when you
swap CUDA for Gaudi.

```
verl-gaudi/
├── README.md                 ← you are here (overview + status)
├── CLAUDE.md                 ← resume instructions for an AI/engineer picking this up
├── docs/
│   ├── 01-verl-pipeline.md   ← 3b1b-style: how verl/GRPO works, first principles
│   ├── 02-gaudi-port.md      ← every Gaudi change mapped to its backend concept (C1–C7)
│   ├── 03-debugging-journey.md ← blow-by-blow: every error, fix, and lesson (ctrl-F your error)
│   ├── memory.md             ← exact project state, checkpoints, next steps
│   └── skills.md             ← the operational playbook (run recipes + cluster gotchas)
├── patches/
│   ├── README.md             ← ordered index of every patch
│   ├── verl05.diff           ← verl 0.5.0 source changes (the active HF-rollout path)
│   ├── verl09_main.diff      ← verl 0.9 source changes (vLLM path)
│   ├── platform_hpu.py       ← the new PlatformHPU backend (verl 0.9)
│   └── env/patch_*.py        ← patches to installed libs (Ray, optimum-habana, enum shim)
└── scripts/                  ← sbatch wrappers + setup/data-prep scripts to reproduce both paths
```

The large binary artifacts (`.sif` containers, model weights, GSM8k parquet, `cpkgs` user-sites) are **not** in git —
they live on Sol at `/scratch/ssamine4/verl_gaudi/`. This repo carries everything needed to *rebuild* them.

---

## Status at a glance

> ✅ **DONE (2026-06-30):** verl GRPO ran **3/3 steps on one Gaudi HPU**, `VERL_RC=0`, all metrics logged.
> Success criterion #3 met. Proof: `logs/SUCCESS_grpo_3steps_57954952.log` (excerpt in `results/`).

| Stage | State |
|-------|-------|
| HPU userspace matches driver (SynapseAI 1.24), real ops run | ✅ |
| verl installed; custom HPU backend; `get_platform()=hpu/intel/8/hccl` | ✅ |
| Ray cluster up on HPU; workers scheduled on `HPU` resources | ✅ (but startup is flaky — see below) |
| FSDP-wraps the actor **on the Gaudi card** (`After FSDP, 3.69 GB on HPU`) | ✅ |
| Reward / advantage / optimizer plumbing | ✅ |
| **Single-card generation (HF rollout)** | ✅ code walls removed (NO_SHARD + ReduceOp coercion) |
| **A full GRPO iteration end-to-end** | ✅ **3/3 steps, `VERL_RC=0`** (Qwen2.5-0.5B GSM8k, 1 HPU, job 57954952) |

### The reframe (session 2): the "walls" were misdiagnosed
Earlier notes treated the HF-rollout path as hitting a fundamental **lazy/eager catch-22**, and concluded the only
way forward was **disaggregated vLLM** (separate HPUs for actor vs. rollout). Fresh root-cause analysis showed that
was wrong:

1. **The "catch-22" was just pointless single-card FSDP.** With one HPU the FSDP mesh is size 1, yet
   `get_sharding_strategy` picked `FULL_SHARD`, whose per-forward `storage._resize_` (free/gather) is the only op
   that needs eager — and it implements *no actual sharding* on one card. **Fix: `NO_SHARD` for a size-1 mesh.** The
   crash vanished and the run advanced to a new, deeper error.
2. **That deeper error: Habana HCCL rejects `ReduceOp.AVG`** (`hccl_kernels.cpp:158`). On a `world_size==1` group
   every collective is an identity, so **coercing `AVG→SUM` is math-exact.** Installed in `verl/utils/device.py`.

**Consequence: the actor can train *and* generate on a single card — disaggregated vLLM is no longer required** to
prove a few iterations.

### What's actually blocking the end-to-end run
A **Ray-in-apptainer metrics-agent startup deadlock** (the project's *original* blocker, resurfaced): the raylet
blocks in `WaitForDashboardAgentPorts()` waiting for the metrics agent, which blocks on a gRPC back to the raylet —
a timing-sensitive deadlock that fires when the agent imports slowly off the shared filesystem. Mitigation in place:
a **cache-warming retry loop** in `_run_inside_05.sh` (attempt 1 warms the page cache so the agent wins the race on
retry). See [`docs/memory.md`](docs/memory.md) for the full mechanism and the next levers if retries are
insufficient (drop the loopback `_node_ip_address`; stage `cpkgs` on node-local disk).

---

## Quick start (on Sol)

```bash
# 0. ASU VPN up, then: ssh sol   (login.sol.rc.asu.edu)
cd /scratch/ssamine4/verl_gaudi

# HF rollout path (single-card train+generate; NO_SHARD + ReduceOp coercion + Ray retry loop):
sbatch scripts/run_05.sh        # -> _run_inside_05.sh
# success markers in slurm-<JID>.out: "After FSDP", "Training Progress", "actor/pg_loss"
# on a failed run, raylet/agent logs are preserved under logs/raylogs_<JID>/

# vLLM path (engine launches; colocation blocked — kept for the disaggregation route):
sbatch scripts/run_vllm.sh      # -> _run_inside_vllm.sh, verl 0.9 + vllm_gaudi
```
Full flag-by-flag explanation of the container invocation and every Hydra override is in
[`docs/skills.md`](docs/skills.md). Rebuilding the containers/env from scratch: `scripts/setup_container_v17.sh`,
`scripts/setup_vllm.sh`, `scripts/prep_data_v20.sh`, `scripts/install_oh18.sh`.

---

## Honest bottom line

> **verl GRPO now runs end-to-end on Intel Gaudi 2.** Qwen2.5-0.5B on GSM8k completed **3/3 steps on a single
> HPU** (`VERL_RC=0`, all metrics logged). The path that earlier notes called fundamentally blocked — needing
> disaggregated vLLM — turned out to be a stack of fixable issues: single-card FSDP `NO_SHARD`, a math-exact
> HCCL `ReduceOp.AVG→SUM` coercion, disabling FSDP mixed precision on HPU (a second lazy-mode resize), an
> `--exclusive` node to acquire a free HPU module, and a cache-warming retry loop around Ray's flaky
> metrics-agent startup. **Disaggregated vLLM was never needed** for the few-iterations goal — the actor trains
> and generates on one card. Caveats are training-quality only (the 0.5B model scores 0 reward on GSM8k at a
> 128-token limit, so `grad_norm` is NaN from zero-advantage cold-start) — orthogonal to the HPU port. All
> patches and environments are checkpointed on `/scratch/ssamine4/verl_gaudi/`.

---

## Credits

Ported and documented on ASU Research Computing (Sol). verl is by the volcengine team; Gaudi software by Intel/Habana.
