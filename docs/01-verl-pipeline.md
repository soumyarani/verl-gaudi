# 01 — How verl Works: LLM Post-Training from First Principles

> Goal of this doc: give you a mental model of **what verl actually does** when it runs GRPO/PPO,
> deep enough that the Gaudi changes in [`02-gaudi-port.md`](02-gaudi-port.md) feel obvious rather than magical.
> Written to be read top-to-bottom like a lecture.

---

## 0. The one-sentence summary

> RL post-training takes a language model that can already *talk* and teaches it to *get things right*,
> by repeatedly: **(1) letting it generate answers, (2) scoring those answers, (3) nudging its weights
> so high-scoring answers become more likely.**

verl ("**HybridFlow**") is an engine that runs this loop efficiently across many GPUs/accelerators.
Everything else is plumbing around those three steps.

---

## 1. The RL post-training loop (the thing we're parallelizing)

After pre-training (next-token prediction on the internet) and SFT (imitating good answers), we do **RLHF/RLVR**.
For math like GSM8k we use **RLVR** — Reinforcement Learning from *Verifiable* Rewards — because we can *check* the answer.

One training step, conceptually:

```
        prompt: "Natalia sold clips to 48 friends... How many in total?"
                         │
                         ▼
   ┌──────────────────────────────────────────────┐
   │ 1. ROLLOUT (generation)                        │
   │    policy model samples N answers per prompt   │   ← this is INFERENCE (autoregressive decode)
   │    e.g. N=4 completions, each up to 128 tokens │
   └──────────────────────────────────────────────┘
                         │  responses
                         ▼
   ┌──────────────────────────────────────────────┐
   │ 2. REWARD                                      │
   │    a verifier scores each answer               │   ← for GSM8k: parse "#### 72", compare to gold
   │    reward = 1.0 if correct else 0.0            │
   └──────────────────────────────────────────────┘
                         │  rewards
                         ▼
   ┌──────────────────────────────────────────────┐
   │ 3. ADVANTAGE                                   │
   │    "was this answer better than average?"      │   ← GRPO: normalize rewards *within the group*
   └──────────────────────────────────────────────┘
                         │  advantages
                         ▼
   ┌──────────────────────────────────────────────┐
   │ 4. POLICY UPDATE (training)                     │
   │    gradient step that raises log-prob of        │   ← this is TRAINING (forward+backward+optimizer)
   │    good tokens, lowers bad ones (+ KL leash)    │
   └──────────────────────────────────────────────┘
                         │
                         ▼  (repeat with next batch)
```

### Why GRPO specifically (and what "advantage" means)

PPO (the classic) needs a **critic** — a second network that predicts "how good is this state?" — to compute the
advantage `A = reward − value`. Training a critic is expensive (another full model).

**GRPO (Group Relative Policy Optimization)** throws the critic away. The trick:
generate a **group** of `N` answers to the *same* prompt, then define each answer's advantage as
**how much better than its groupmates** it was:

```
A_i = (r_i − mean(r_1..r_N)) / std(r_1..r_N)
```

That's why the config has `actor_rollout_ref.rollout.n = 4`: 4 samples per prompt form one group.
No critic network → less memory, simpler. This is why our run **disabled the critic**
(`Disabled critic as algorithm.adv_estimator != gae`).

### The four "models" in the loop (for GRPO, 3 are active)

| Role        | What it is                          | Used in step | Trained? |
|-------------|-------------------------------------|--------------|----------|
| **Actor**   | the policy we're improving          | 1 (gen) + 4 (update) | yes |
| **Rollout** | a fast inference copy of the actor  | 1 (gen)      | no (reads actor weights) |
| **Ref**     | a *frozen* copy of the start model  | 4 (KL term)  | no (frozen) |
| ~~Critic~~  | value network                       | (PPO only)   | (disabled for GRPO) |

The **KL term** (ref model) is a leash: it penalizes the actor for drifting too far from the original model,
so RL doesn't destroy the language ability the model already had.

**Key insight that drives the whole engine design:** the Actor and the Rollout are *the same weights* but used in
two totally different compute regimes — **training** (big batched forward+backward, needs gradients, FSDP-sharded)
vs **inference** (autoregressive token-by-token decode, no gradients, wants a fast engine like vLLM).
verl's whole job is to make these two coexist on the same hardware. **This coexistence is exactly what broke on Gaudi.**

---

## 2. verl's architecture: single-controller + workers

verl is built on **Ray** (a distributed Python framework). The design is "**single controller, many workers**":

```
                      ┌─────────────────────────────────────┐
                      │  DRIVER  (one process, the "brain")  │
                      │  RayPPOTrainer.fit()                 │
                      │  - holds the dataloader              │
                      │  - orchestrates the 4 steps          │
                      │  - computes advantages, metrics      │
                      └─────────────────────────────────────┘
                          │ ray.remote calls (RPC)   ▲ results (DataProto)
                          ▼                          │
   ┌───────────────────────────────────────────────────────────────┐
   │  WORKERS (Ray actors, one per GPU/HPU)                          │
   │  WorkerDict = { actor_rollout: ActorRolloutRefWorker, ... }     │
   │   ├─ FSDP-wrapped policy model (training)                       │
   │   ├─ rollout engine (generation): hf | vllm | sglang            │
   │   └─ ref model (KL)                                             │
   └───────────────────────────────────────────────────────────────┘
```

- The **driver** runs `main_ppo.py -> run_ppo() -> ray.init() -> RayPPOTrainer.fit()`. It never touches a GPU
  directly; it sends *batches* to workers and gets *batches* back.
- The **workers** are Ray actors pinned to accelerators. Each holds the FSDP model + a rollout engine.
- Data moves between them as **`DataProto`** objects (verl's batch container: a `TensorDict` of tensors +
  a dict of non-tensor metadata like `uid`, `eos_token_id`). When you saw
  `AssertionError: Two tensor dict must have identical batch size. Got 64 and 256`, that was a `DataProto.union`
  of mismatched batches.

### Resource pools & "colocation" (this is where Gaudi fought us)

With `n_gpus_per_node=1`, verl puts the actor, rollout, and ref **all on the same 1 accelerator** — "colocation" /
"hybrid engine". On CUDA this is free: multiple processes/contexts share a GPU. verl requests **fractional**
resources (`1/3 GPU` each) so Ray will schedule three colocated roles on one device.

> **Foreshadowing:** Gaudi (Habana) modules are **exclusive per process** — you cannot have the FSDP actor
> *and* a separate vLLM server process both hold module 0. And Ray refuses *fractional* `HPU` quantities by default.
> Two of our ~20 patches exist purely because of this CUDA-assumption.

---

## 3. One GRPO step in code (the path our run actually took)

This is `RayPPOTrainer.fit()` in `verl/trainer/ppo/ray_trainer.py`. Annotated with where things broke on Gaudi:

```python
for batch in dataloader:                          # batch of 16 prompts
    gen_batch = batch.pop(gen_keys)               # just the prompt tensors

    # -- repeat each prompt n=4 times so we get a "group" per prompt --
    gen_batch = gen_batch.repeat(rollout.n, interleave=True)   # 16 -> 64

    # -- STEP 1: ROLLOUT (generation) --
    gen_out = actor_rollout_wg.generate_sequences(gen_batch)   # <-- THE WALL: HPU generation
        # -> worker -> self.rollout.generate_sequences()
        #   hf rollout:   FSDP.summon_full_params() then model.generate()  (optimum-habana / lazy-vs-eager hell)
        #   vllm rollout: a separate vLLM server process                   (can't acquire the Gaudi module)

    batch = batch.repeat(rollout.n, interleave=True)           # align the 64
    batch = batch.union(gen_out)                               # batch-size asserts live here

    # -- STEP 2: REWARD --
    reward = reward_fn(batch)                      # GSM8k: parse "#### N", compare to gold -> 1.0/0.0

    # -- ref + old log-probs (for the KL leash & importance ratio) --
    batch = batch.union(ref_policy_wg.compute_ref_log_prob(batch))
    batch = batch.union(actor_rollout_wg.compute_log_prob(batch))

    # -- STEP 3: ADVANTAGE (GRPO group-normalization) --
    batch = compute_advantage(batch, adv_estimator="grpo")

    # -- STEP 4: POLICY UPDATE --
    actor_rollout_wg.update_actor(batch)           # forward + backward + optimizer.step on FSDP model
```

Notice generation (step 1) is **half the loop** and is *inference*, while step 4 is *training*. The engine has to
flip the same weights between these two modes every step. That flip is **weight resync** + **memory juggling**
(offload the trainer, wake the inference engine, sleep it again). On HPU, every one of those primitives needed work.

---

## 4. The two compute regimes, precisely

### Training side: FSDP

The actor model is wrapped in **FSDP** (Fully Sharded Data Parallel). FSDP shards parameters/grads/optimizer-state
across the data-parallel group, gathering full params *just-in-time* for each layer's forward/backward, then
re-sharding. Key FSDP operations that matter later:

- **`summon_full_params(model)`** — temporarily gather *all* shards into full params (used so a non-FSDP code path,
  like `.generate()`, can see whole weights). Internally it allocates full params and **frees** the shard storage
  via `storage._resize_(0)`.   *That `_resize_(0)` is unsupported in Habana lazy mode -> our catch-22.*
- **device placement** — FSDP, when given `device_id=`, **moves** the (CPU/meta) module to the device during
  wrapping using `tensor.set_data(...)`.   *`set_data` across CPU->HPU storage types is rejected on Gaudi.*
- **mixed precision** — params kept in fp32, cast to bf16 for compute.   *Interacts badly with optimum-habana's
  fp32 static KV cache -> an `index_copy_` dtype clash.*

### Inference side: the rollout engine

verl supports several rollout backends, registered by name:

- **`hf`** — plain `transformers` `model.generate()` on the *training* model (FSDP), in-process. Slow but simple.
  *This is the path that actually generated tokens on Gaudi.* (Note: verl >=0.9 **removed** this from its new
  engine-worker path — it only exists in the legacy `fsdp_workers.py`, which is why we used **verl 0.5.0** for it.)
- **`vllm`** — a high-throughput inference server (PagedAttention, continuous batching). Runs as a **separate
  process/server** that verl syncs weights into each step. *This is the "real" path — but its separate process
  can't share a Gaudi module with the FSDP actor.*
- **`sglang`**, **`trtllm`** — other servers (no Gaudi support).

The reason the rollout is a *separate fast engine* (vLLM) instead of just `model.generate()` is throughput:
autoregressive decode of thousands of tokens dominates RL wall-clock, and vLLM is ~10-20x faster than HF generate.
That speed is why everyone uses vLLM — and why "verl on Gaudi" really means "vLLM-gaudi on Gaudi."

---

## 5. Where this maps to the run logs you saw

| Log line / error                                   | Which part of the loop |
|----------------------------------------------------|------------------------|
| `Connected to Ray cluster` / `0.0/8.0 HPU`         | section 2 driver/workers, resource pool |
| `After FSDP, memory allocated 3.69 GB`             | section 4 training side, FSDP wrapped the actor on the HPU |
| `Total training steps: 3 / Training from scratch`  | section 3 `fit()` loop entered |
| `actor_rollout_generate_sequences()`               | section 3 step 1 — generation |
| `summon_full_params -> storage._resize_ -> lazy flow`| section 4 FSDP gathering full weights for `hf` generate |
| `Device acquire failed` (vLLM worker)              | section 1/4 vLLM's separate process can't grab the Gaudi module |
| `index_copy_ Float vs BFloat16`                    | section 4 mixed-precision vs optimum-habana fp32 cache |
| `Two tensor dict must have identical batch size`   | section 3 `batch.union(gen_out)` with double-applied `n` |

Read [`02-gaudi-port.md`](02-gaudi-port.md) next: it takes each of these and shows the exact line we changed and the
backend concept behind it.
