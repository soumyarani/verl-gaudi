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
1. [`docs/memory.md`](docs/memory.md) — **current state, the two walls, the recommended next step, checkpoint map.**
2. [`docs/skills.md`](docs/skills.md) — the run recipes and the cluster gotchas (sbatch-only, `--cleanenv`, env vars).
3. [`docs/01-verl-pipeline.md`](docs/01-verl-pipeline.md) — what verl/GRPO actually does (so the patches make sense).
4. [`docs/02-gaudi-port.md`](docs/02-gaudi-port.md) — each change mapped to its backend concept.
5. [`docs/03-debugging-journey.md`](docs/03-debugging-journey.md) — ctrl-F any error message you hit.

## The environment (don't rediscover this the hard way)
- Host: `ssh sol` (`login.sol.rc.asu.edu`, user `ssamine4`), **requires ASU VPN**. If `sol` won't resolve, the VPN
  dropped — that's a human action, not something you can fix; say so.
- **`sbatch` only** (no interactive `srun`/`salloc`). Run on a **`gaudi` node** (login node can't import HPU torch).
- Everything is checkpointed at `/scratch/ssamine4/verl_gaudi/` — two `.sif` containers, two verl clones
  (`verl/`=0.9 vLLM path, `verl05/`=0.5.0 HF path), two user-site dirs (`cpkgs`, `cpkgs_vllm`), model, data, scripts,
  logs. The `patches/` and `scripts/` in *this repo* mirror the source/env changes.

## What "done" looks like
A full GRPO iteration completes: generation → reward → advantage → `update_actor`, for a few steps, with the loss
logged. Today it stops inside **generation**. The single highest-value task is **disaggregated vLLM** (actor and
vLLM rollout on *separate* HPUs) — see `docs/memory.md` "Recommended next step". Sanity-check a standalone
`vllm_gaudi` serve of Qwen2.5-0.5B in the vLLM container first, then wire verl's non-colocated rollout placement.

## When you change verl or an installed lib
- verl source changes → keep `patches/verl05.diff` / `patches/verl09_main.diff` regenerated (`git -C <clone> diff`).
- Installed-lib changes (Ray, optimum-habana) live in `patches/env/patch_*.py` and must be **re-applied after any
  reinstall** of those libs. Document any new one in `patches/README.md`.
- Always reproduce via `sbatch`, read the log under `/scratch/ssamine4/verl_gaudi/logs/`, and restate the result in
  your own words (tool output isn't visible to the task tracker).

## Don't
- Don't try to fix the VPN, or use interactive `srun`. Don't run HPU torch on the login node.
- Don't colocate a second HPU process on the actor's module and expect it to work — that's wall #1, it's by design.
- Don't bump transformers/optimum-habana/torch casually — they're pinned to match the 1.24 driver; pins are in
  `cconstraints.txt` / `cv_constraints.txt`.
