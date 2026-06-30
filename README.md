# verl-gaudi

Running **[verl](https://github.com/volcengine/verl)** (the HybridFlow RL post-training framework) on **Intel
Gaudi 2 (HPU)** accelerators — the full port, every patch, both container environments, and a from-first-principles
explanation of *why* each change was needed.

Target run: GRPO post-training of **Qwen2.5-0.5B-Instruct** on **GSM8k**, on the ASU **Sol** supercomputer's
`gaudi` partition (8× HL-225, SynapseAI 1.24).

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
│   ├── verl05.diff           ← verl 0.5.0 source changes (the path that generated tokens)
│   ├── verl09_main.diff      ← verl 0.9 source changes (vLLM path)
│   ├── platform_hpu.py       ← the new PlatformHPU backend (verl 0.9)
│   └── env/patch_*.py        ← patches to installed libs (Ray, optimum-habana, enum shim)
└── scripts/                  ← sbatch wrappers + setup/data-prep scripts to reproduce both paths
```

The large binary artifacts (`.sif` containers, model weights, GSM8k parquet, `cpkgs` user-sites) are **not** in git —
they live on Sol at `/scratch/ssamine4/verl_gaudi/`. This repo carries everything needed to *rebuild* them.

---

## Status at a glance

| Stage | State |
|-------|-------|
| HPU userspace matches driver (SynapseAI 1.24), real ops run | ✅ |
| verl installed; custom `PlatformHPU` backend; `get_platform()=hpu/intel/8/hccl` | ✅ |
| Ray cluster up on HPU; workers scheduled on `HPU` resources | ✅ |
| FSDP-wraps the actor **on the Gaudi card** (`After FSDP, 3.69 GB on HPU`) | ✅ |
| Reward / advantage / optimizer plumbing | ✅ |
| **Generation (rollout)** | ⚠️ ran ~36 s on HPU (HF path) then hit a wall |
| **A full GRPO iteration end-to-end** | 🧱 not yet — see below |

Two architectures were tried; each hits one HPU-specific wall:
1. **vLLM rollout** (verl 0.9): vLLM launches fine on HPU, but as a *separate process* it can't acquire the Gaudi
   module the FSDP actor already holds (Gaudi modules are exclusive-per-process).
2. **HF rollout** (verl 0.5.0 + optimum-habana): runs `generate()` through the FSDP graph → FSDP's
   `summon_full_params` needs **eager** mode while optimum-habana's `generate` needs **lazy** — a true catch-22.

Both point to the same fix: **disaggregated vLLM** — put the actor and the vLLM rollout server on *different* HPUs.
See [`docs/memory.md`](docs/memory.md) for the concrete next steps.

---

## Quick start (on Sol)

```bash
# 0. ASU VPN up, then: ssh sol   (login.sol.rc.asu.edu)
cd /scratch/ssamine4/verl_gaudi

# HF rollout path (generates tokens):
sbatch scripts/run_05.sh        # -> _run_inside_05.sh, verl05 + optimum-habana, eager

# vLLM path (engine launches; colocation blocked):
sbatch scripts/run_vllm.sh      # -> _run_inside_vllm.sh, verl 0.9 + vllm_gaudi
```
Full flag-by-flag explanation of the container invocation and every Hydra override is in
[`docs/skills.md`](docs/skills.md). Rebuilding the containers/env from scratch: `scripts/setup_container_v17.sh`,
`scripts/setup_vllm.sh`, `scripts/prep_data_v20.sh`, `scripts/install_oh18.sh`.

---

## Honest bottom line

> verl is installed and runs on Gaudi through the entire training pipeline up to and including generation — but the
> rollout/generation step hits HPU-specific op-compatibility walls in both available architectures (vLLM
> colocation; HF-generate+FSDP+optimum-habana). Completing a full iteration needs one of: disaggregated vLLM
> (dedicate separate HPUs to actor vs. rollout), a Habana-validated optimum-habana/transformers/verl pin for this
> exact path, or upstream fixes. All ~20 patches and both environments are checkpointed on
> `/scratch/ssamine4/verl_gaudi/`.

---

## Credits

Ported and documented on ASU Research Computing (Sol). verl is by the volcengine team; Gaudi software by Intel/Habana.
