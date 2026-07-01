# Benchmark: verl GRPO on Gaudi 2 vs A100 (Qwen2.5-0.5B, GSM8k)

WandB project: **https://wandb.ai/ssamine4-arizona-state-university/verl-gaudi-vs-a100**

All runs: 1 accelerator, Qwen2.5-0.5B-Instruct, GSM8k, GRPO, verl 0.5.0. Two regimes — a **matched** config
(identical code path on both, to isolate hardware) and **verl's full reference settings** (how you'd really train).

---

## Headline

| Regime | Gaudi 2 (HL-225, 92 GB) | A100 (80 GB) | Notes |
|--------|--------------------------|--------------|-------|
| **Matched** (HF rollout, fp32, sdpa, bs=16, resp=128, n=4) | **10.2 s/step** | **6.2 s/step** | A100 **~1.6×** faster, same code path |
| **Full verl settings** (bs=1024, resp=1024, n=5, KL loss) | **>38 min/step — did not complete** (HF rollout) | **~570 s/step (9.5 min)** — completes, *learns* (native vLLM) | The rollout engine, not FLOPS, decides feasibility |

**The one-line takeaway:** on the *same* small workload A100 is ~1.6× a Gaudi 2. But at *real* settings the gap
explodes — not because of raw compute, but because A100 runs verl's **vLLM** rollout (batched/paged generation)
while Gaudi is stuck on the **HF** rollout (serial `model.generate`), which never finished one step. **This is the
concrete, measured argument for why a Gaudi port needs `vllm-gaudi`.**

---

## Matched config (apples-to-apples, isolates hardware)

Identical on both: `rollout.name=hf`, fp32 + `mixed_precision=None`, `attn=sdpa`, `NO_SHARD`, `n=4`,
`train_batch_size=16`, `max_prompt=256`, `max_response=128`, `lr=1e-6`. Both exited `VERL_RC=0`.

| Metric | Gaudi 2 | A100 |
|--------|---------|------|
| Steady-state per step | **10.22 s** (mean over 96 steps) | **~6.2 s** |
| Step 1 (graph compile) | 240.8 s | 11.1 s |
| Generation (steady) | ~5.3 s | ~3.0 s |
| Update-actor (steady) | ~4.3 s | ~2.7 s |
| 100-step total wall | **23 min 07 s** (`VERL_RC=0`) | — (ran 3 steps) |
| Reward | 0.0 (0.5B too weak at 128-tok) | 0.0 |

Notes: Gaudi's step 1 is dominated by SynapseAI **graph compilation** (240 s), which then amortizes to ~10 s — a
real Gaudi characteristic. Both score 0 reward here: 128-token responses are too short for GSM8k chain-of-thought.
(WandB: Gaudi run `kywjzevs`.)

---

## Full verl settings (verl's reference GSM8k GRPO)

`train_batch_size=1024`, `max_prompt=512`, `max_response=1024`, `n=5`, `use_kl_loss=True (low_var_kl, 0.001)`,
`ppo_mini_batch=256`. This is the config from verl's quickstart — designed for vLLM.

### A100 — native verl + vLLM (`VERL_RC=0`, the intended path)
Unmodified verl 0.5.0, `rollout.name=vllm`, flash-attention, installed with **uv** (`uv pip install vllm==0.8.5`).

| Step | gen | old_log_prob | ref | update_actor | **total** | reward (mean) |
|------|-----|--------------|-----|--------------|-----------|---------------|
| 1 | 110.6 s | 118.2 s | 82.4 s | 265.6 s | **580.4 s** | 0.011 |
| 2 | 114.4 s | 108.0 s | 80.9 s | 260.1 s | **566.8 s** | 0.022 |

- **~9.5 min/step**, ~3,800 tok/s, ~10.6% MFU, peak 46.6 GB.
- **It actually learns:** reward climbs 0.011→0.022 (max 1.0 — some GSM8k solved), advantages/grad_norm finite,
  because 1024-token responses fit real chain-of-thought (mean response 323 tokens).
- Note where time goes: generation is only ~20% of the step; **update_actor (forward+backward on 5,120×~430
  tokens) dominates**. vLLM makes generation cheap; the actor update is the new bottleneck. (WandB: `mfut063z`.)

### Gaudi 2 — HF rollout (`rollout.name=hf`, the only option on Gaudi today)
verl 0.9 removed the HF rollout; vLLM colocation is blocked on Gaudi — so full settings must use the legacy HF
rollout. It **did not complete a single step**: generation ran **>38 minutes and was still going** before we
stopped it. Root cause: the HF rollout has **no batched/paged generation** — it materializes 5,120 sequences in 80
serial chunks of 64, each up to 1,024 tokens, with HPU graph recompilation per shape. A first-attempt OOM
(single 46 GB allocation) also had to be worked around with `rollout.micro_batch_size=64`.

Getting even this far required fixing **four previously-untested Gaudi port gaps** in the KL/reference-model path
(never exercised by the matched run, which had no KL): ref FSDP wrap-policy can't find the optimum-habana-renamed
`GaudiQwen2DecoderLayer`; two CPU↔HPU `set_data`/`CPUOffload` conflicts in the ref build; and the generation
micro-batch OOM. All are documented as material for the HPU plugin.

---

## Honest caveats

- **The full-settings row is not apples-to-apples** (Gaudi HF+sdpa vs A100 vLLM+flash) — but that's the point the
  user asked to see: *native verl on A100* vs *the ported path on Gaudi*. The matched row is the controlled
  hardware comparison.
- Gaudi numbers are **fp32** (HPU lazy mode can't do FSDP mixed precision — see the port notes); A100 native uses
  its default precision. This favors A100 on the matched row too, but reflects the real state of the port.
- Single-card, tiny model — not a throughput-optimized or multi-node result. It answers "does verl run, and how
  fast per step" for the few-iterations goal, not "what's peak training throughput."

## Reproduce
- Gaudi matched 100-step: `sbatch scripts/bench_gaudi_v01.sh` (logs `logs/BENCH_gaudi_100steps_*.log`).
- A100 native vLLM full: `sbatch scripts/bench_a100_vllm_v01.sh` (uv venv, `logs/BENCH_a100_vllm_FULL_*.log`).
- A100 matched (HF rollout): `scripts/bench_a100_v01.sh` (NGC container `cpkgs_cuda`).
