# CLAUDE.md — instructions for picking this project up

You (Claude, or a human) are resuming the **verl-on-Gaudi** port. This file tells you how to behave and where to
look. Read it fully before touching anything.

## How to work (carried from the user's standing rules)
1. **Think before coding.** State assumptions; surface tradeoffs; if multiple interpretations exist, present them —
   don't silently pick. If something's unclear, stop and ask.
2. **Simplicity first.** Minimum code that solves the problem. No speculative abstractions or config.
3. **Surgical changes.** Touch only what the task needs; match existing style; don't refactor what isn't broken.
   Every changed line should trace to the request. Mention unrelated dead code — don't delete it.
4. **File locations.** Never write to `/tmp` or scratch dirs the user hasn't approved. Propose a path and wait.
   Version standalone scripts `<name>_vNN.ext` (zero-padded; highest = latest) — create the next version, don't
   silently overwrite. Cross-linked HTML keeps stable names.
5. **Goal-driven.** Turn each task into a verifiable check and loop until it passes.

## Orientation — read in this order
1. [`docs/memory.md`](docs/memory.md) — **current state, the reframe (NO_SHARD + ReduceOp), the live Ray blocker,
   the checkpoint map.** Start here.
2. [`docs/skills.md`](docs/skills.md) — the run recipes and the cluster gotchas (sbatch-only, `--cleanenv`, env vars,
   the Ray retry loop, the NO_SHARD/ReduceOp/agent-deadlock notes at the end).
3. [`docs/01-verl-pipeline.md`](docs/01-verl-pipeline.md) — what verl/GRPO actually does (so the patches make sense).
4. [`docs/02-gaudi-port.md`](docs/02-gaudi-port.md) — each change mapped to its backend concept.
5. [`docs/03-debugging-journey.md`](docs/03-debugging-journey.md) — ctrl-F any error message you hit. **Phase 7** at
   the end is the most current (the fresh-eyes pass that removed the "walls").

## The environment (don't rediscover this the hard way)
- Host: `ssh sol` (`login.sol.rc.asu.edu`, user `ssamine4`), **requires ASU VPN**. If `sol` won't resolve, the VPN
  dropped — that's a human action, not something you can fix; say so.
- **`sbatch` only** (no interactive `srun`/`salloc`). Run on a **`gaudi` node** (login node can't import HPU torch).
- A persistent **SSH ControlMaster** avoids re-Duo on every command:
  `ControlMaster auto` + `ControlPath ~/.ssh/cm/%r@%h:%p` + `ControlPersist 30m`; open it once, reuse silently.
- Everything is checkpointed at `/scratch/ssamine4/verl_gaudi/` — two `.sif` containers, two verl clones
  (`verl05/`=0.5.0 **active HF path**, `verl/`=0.9 vLLM path), user-site dirs (`cpkgs`, `cpkgs_vllm`), model, data,
  scripts, logs. The `patches/` and `scripts/` in *this repo* mirror the source/env changes.

## What "done" looks like — and the current obstacle
A full GRPO iteration completes: generation → reward → advantage → `update_actor`, for a few steps, with the loss
logged. **The code walls are removed** (single-card `NO_SHARD` + `ReduceOp.AVG→SUM` coercion mean the actor can
train *and* generate on one card — **disaggregated vLLM is NOT needed** for the few-iterations goal). The one thing
left is **infrastructure**: a **Ray-in-apptainer metrics-agent startup deadlock**
(`WaitForDashboardAgentPorts` ↔ agent gRPC). The `_run_inside_05.sh` **cache-warming retry loop** is the current
mitigation. If it's still flaky, the next levers (in `docs/memory.md`) are: drop `_node_ip_address="127.0.0.1"` in
`main_ppo.py`; stage `cpkgs` on node-local disk; or pre-start a head node. **Only fall back to disaggregated vLLM if
those fail.**

To verify a run: `sbatch scripts/run_05.sh`, then watch `slurm-<JID>.out` for `After FSDP` → `Training Progress` →
`actor/pg_loss`. On failure, read `logs/raylogs_<JID>/raylet.err` for the *real* reason.

## When you change verl or an installed lib
- verl source changes → regenerate the diff: `git -C /scratch/ssamine4/verl_gaudi/verl05 diff > patches/verl05.diff`.
- Installed-lib changes (Ray, optimum-habana) live in `patches/env/patch_*.py` and must be **re-applied after any
  reinstall**. Document any new one in `patches/README.md`.
- The **ReduceOp coercion belongs in `verl/utils/device.py`**, NOT a global `usercustomize.py` — a global hook drags
  `import torch` into every Ray helper process and breaks raylet startup.
- Always reproduce via `sbatch`, read the log, and restate the result in your own words (tool output isn't visible
  to the task tracker).

## Don't
- Don't try to fix the VPN, or use interactive `srun`. Don't run HPU torch on the login node.
- Don't use `FULL_SHARD` on a single card — it emits `storage._resize_` ops HPU lazy mode can't do. `NO_SHARD` for a
  size-1 mesh.
- Don't put the ReduceOp coercion in a global `usercustomize.py` (breaks Ray helper startup).
- Don't keep blindly switching nodes when Ray fails — read `raylogs_<JID>/raylet.err` first; the cause is usually the
  metrics-agent deadlock (retry it) or a stuck HPU module (`Device acquire failed` → `--exclude` that node).
- Don't bump transformers/optimum-habana/torch casually — pinned to the 1.24 driver (`cconstraints.txt`).
