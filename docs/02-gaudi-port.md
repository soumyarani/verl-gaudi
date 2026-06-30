# 02 — Porting verl to Intel Gaudi: Every Change, and the Backend Concept Behind It

> Read [`01-verl-pipeline.md`](01-verl-pipeline.md) first. This doc is the "PyTorch backend masterclass":
> each change we made is a window into how PyTorch talks to an accelerator that **isn't** CUDA.

The actual code is in [`../patches/`](../patches/). Diffs: `verl05.diff` (the path that ran generation),
`verl09_main.diff` (the vLLM path + the platform plugin), `platform_hpu.py` (new backend), and `env/*.py`
(patches to *installed* libraries: Ray, optimum-habana — these live outside verl's git tree).

---

## A. Background: what a Gaudi actually is, in PyTorch terms

A GPU in PyTorch is reached through the **CUDA backend**: `torch.cuda`, tensors with `device='cuda'`, NCCL for
collectives. Intel **Gaudi** (a.k.a. **HPU** — Habana Processing Unit) is a *different* accelerator with its own stack:

| Concept                 | NVIDIA (CUDA)            | Intel Gaudi (HPU)                          |
|-------------------------|-------------------------|--------------------------------------------|
| PyTorch device          | `torch.cuda`, `'cuda'`  | `torch.hpu`, `'hpu'` (added by a plugin)   |
| Runtime / driver        | CUDA + cuDNN            | **SynapseAI** (`/opt/habanalabs`, driver 1.24) |
| PyTorch integration     | built into PyTorch      | **`habana_frameworks.torch`** (a plugin you `import`) |
| Collectives (multi-dev) | NCCL                    | **HCCL** (`"hccl"` backend)                |
| Visible-device env var  | `CUDA_VISIBLE_DEVICES`  | `HABANA_VISIBLE_MODULES`                   |
| Kernels                 | compiled CUDA           | **TPC** kernels + a **graph compiler** (`GC_KERNEL_PATH`) |
| Fast inference engine   | vLLM (built-in)         | **vllm-gaudi** plugin (registers an HPU platform in vLLM) |
| HF `generate()` support | works out of the box    | needs **optimum-habana** adaptations       |

Two facts about Gaudi drive *most* of the pain:

1. **Lazy vs. Eager execution.** By default Gaudi runs in **lazy mode**: ops are recorded into a graph and only
   compiled+executed at a `mark_step()` (or when you read a value). This is great for throughput but means some
   ops (storage resize, certain views, control flow) "should not be called in lazy flow." **Eager mode**
   (`PT_HPU_LAZY_MODE=0`) executes op-by-op like CUDA, but its graph compiler rejects some *other* ops (e.g.
   `strided_view`). Neither mode supports the full set vanilla code assumes → we ping-ponged between them.
2. **Exclusive device acquisition.** A process `acquire`s a Gaudi *module*; a second process trying to grab the
   same module gets `synStatus=8 [Device not found] Device acquire failed`. CUDA contexts can share a GPU; Gaudi
   modules basically can't. This single fact killed the colocated-vLLM architecture.

---

## B. The environment (before any code change)

The hardest *non-code* lesson: **the software stack must match the driver version, and Habana ships it in a
container, not as pip wheels.**

- ASU Sol's prebuilt conda envs were SynapseAI **1.22/1.23** but the nodes run driver **1.24** → even `a + a` on an
  HPU failed to graph-compile (`synStatus 26`, a 1.24 ComplexGUID lib loaded into a 1.22 bridge). **Mismatched
  userspace vs driver = nothing works.**
- Fix: run inside Habana's official **1.24 container** via Apptainer/Singularity (`vault.habana.ai/.../pytorch-installer-2.10.0`
  for the FSDP path; `.../vllm-...-ptfork-2.10.0` for the vLLM path — that one bundles `vllm 0.9.1 + vllm_gaudi`).
- Inside the container, verl + deps go into a **writable user-site on scratch** (`PYTHONUSERBASE=$WS/cpkgs`) because
  the container's `site-packages` is read-only (squashfs).
- The "magic" run incantation (see `scripts/run_05.sh`) and why each flag exists:
  - `--cleanenv --no-home` — Sol's Apptainer binds the host `~/.local` and Lmod modules over the container,
    which shadowed the container's HPU torch with a stray **CUDA** torch. These flags stop that.
  - `--env GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so` etc. — the graph compiler needs the TPC kernel
    library path; the container only sets it via a login shell, so we set it explicitly.
  - `--env HABANA_LOGS=<writable>` — SynapseAI aborts init if it can't write its logs (`/var/log` is read-only).
  - `/usr/bin/python3.10 -s` + `PYTHONNOUSERSITE` discipline — keep the container's HPU torch, not host packages.

> Takeaway for any non-CUDA accelerator: **version-match the whole stack to the driver, isolate the environment
> aggressively, and verify a trivial op (`(x@x).sum()`) on the device before touching the framework.**

---

## C. The code changes, grouped by the concept they teach

### C1. Teaching PyTorch/verl that "hpu" exists  →  `verl/utils/device.py` (+ `platform_hpu.py` for v0.9)

verl abstracts the device behind helpers: `get_device_name()`, `get_torch_device()`, `get_nccl_backend()`,
`get_visible_devices_keyword()`. Out of the box they only know `cuda` and `npu`. We taught them `hpu`:

```python
def is_torch_hpu_available():
    import habana_frameworks.torch   # importing this is what *registers* torch.hpu and the "hccl" backend
    return hasattr(torch, "hpu") and torch.hpu.is_available()

# get_device_name():       ... elif is_hpu_available: device = "hpu"
# get_nccl_backend():      ... elif is_npu_available or is_hpu_available: return "hccl"
# get_visible_devices...:  ... if is_hpu_available: return "HABANA_VISIBLE_MODULES"
```

**Concept:** PyTorch device support is a *registry*. `import habana_frameworks.torch` injects `torch.hpu` and an
`"hccl"` distributed backend at import time — the same way `import torch_npu` injects `torch.npu`. verl's whole
codebase then flows through `get_torch_device()` (which is just `getattr(torch, "hpu")`), so one detection point
lights up hundreds of call-sites.

verl **0.9** has a fancier `verl/plugin/platform/` registry instead of a flat `device.py`. We added
[`platform_hpu.py`](../patches/platform_hpu.py) — a `PlatformHPU(PlatformBase)` mirroring the Ascend-NPU plugin:
`device_name="hpu"`, `communication_backend_name()="hccl"`, `ray_resource_name()="HPU"`, etc. — and registered it.
**Same idea, newer plumbing.**

### C2. Ray doesn't know HPUs are accelerators  →  `single_controller/ray/base.py`, `ray_trainer.py`, Ray itself

Three separate Ray-isms assumed CUDA:

1. **Worker placement.** verl asks Ray for `num_gpus` (CUDA) or `{"resources": {"NPU": n}}` (Ascend). HPU matched
   neither → workers got 0 accelerators. We added the `hpu -> {"resources": {"HPU": n}}` branch and made the
   placement-group bundle use `"HPU"`.
2. **Resource accounting.** `_check_resource_available()` counted only `GPU` or `NPU` per node → "Total available
   GPUs 0 < 1". We extended it to also count `HPU`. *(Ray auto-detects HPUs and even exposes an `HPU` custom
   resource — but only if you start it right; see C3.)*
3. **Fractional resources.** verl colocates actor+rollout+ref by requesting **⅓ HPU each**. Ray *forbids fractional
   accelerator quantities* for `HPU/NPU/TPU` (`ray_option_utils.py`: "HPU resource quantity must be whole numbers").
   Multiple processes *can* share a Gaudi for colocation, so we patched Ray to allow fractional HPU
   ([`env/patch_ray.py`](../patches/env/patch_ray.py)).

**Concept:** A cluster scheduler models accelerators as **named resources**. CUDA GPUs get the privileged `num_gpus`
field (fractional allowed, env vars auto-set); everything else is a generic custom resource with stricter rules.
Porting a CUDA-first framework means finding every `"GPU"`/`num_gpus` assumption and teaching it your resource name.

### C3. Making Ray itself start on Gaudi (the multi-hour boss fight)  →  run scripts + env

Pure infrastructure, no verl code, but the single biggest time sink. Symptoms and root causes:

- **`ray start` raylet crash**: `node_manager.cc Check failed ... Timed out waiting for metrics_agent_port`.
  Root cause: **`RAY_agent_register_timeout_ms` default is too short**; the dashboard/metrics agent imports a huge
  dependency tree off slow beegfs and can't register in time → the raylet *fatally* crashes. Fix:
  `RAY_agent_register_timeout_ms=300000`.
- **Gaudi mis-detected as `TPU`**: with the agent fixed, `ray status` showed `0.0/1.0 TPU` instead of `8 HPU`.
  Fix: declare the resource explicitly, `ray.init(resources={"HPU": 8})` (or `ray start --resources='{"HPU":8}'`),
  plus `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`.
- **`ray` CLI itself broken** in the vLLM image (`is not a valid Sentinel`, a dep-version skew) → we **bypassed the
  CLI** and let verl's own in-process `ray.init(...)` start the head, passing
  `+ray_kwargs.ray_init.resources={HPU:8} +ray_kwargs.ray_init._node_ip_address=127.0.0.1`.

**Concept:** Ray's per-node "dashboard agent" is a *separate process* the raylet waits for, on a hard timeout, in a
graph-FS + container environment those defaults are wrong. And Ray's accelerator auto-detection is heuristic — when
in doubt, **declare your resources explicitly** rather than trusting detection.

### C4. Attention kernels: FlashAttention is CUDA-only  →  `fsdp_workers.py`, `engine.py`, `model.py`

verl loads models with `attn_implementation="flash_attention_2"`. `flash_attn` is a **CUDA-only** package; on HPU
it isn't installed and would be the wrong kernel anyway. We switched the defaults to `"sdpa"` (PyTorch's
`scaled_dot_product_attention`, which Habana implements with a fused HPU kernel).

**Concept:** "attention implementation" is a pluggable kernel choice in `transformers`. `flash_attention_2` →
the FlashAttention CUDA kernel; `sdpa` → `torch.nn.functional.scaled_dot_product_attention`, which each backend
(CUDA, HPU, CPU) provides its own fused version of. Portable code picks `sdpa`. (Also why `use_remove_padding=False`:
that path relies on flash-attn's variable-length kernel.)

### C5. FSDP moving weights onto the device  →  `fsdp_workers.py` (the "pre-move" patch)

```python
if fsdp_strategy == "fsdp":
    if get_device_name() == "hpu":
        actor_module = actor_module.to("hpu")   # <-- our 2-line fix
    actor_module_fsdp = FSDP(actor_module, device_id=get_device_id(), ...)
```

Without this: `RuntimeError: Attempted to call variable.set_data(tensor), but variable and tensor have incompatible
tensor type`. FSDP, given `device_id=`, moves a CPU module to the device *inside* the constructor via
`tensor.set_data(...)`. On Gaudi, swapping a **CPU storage** for an **HPU storage** via `set_data` is rejected
(different storage types). If the module is *already* on HPU, FSDP's move is a no-op and the problem vanishes.

**Concept:** A PyTorch tensor = (metadata) + (a pointer to **Storage**, which lives on a device). `set_data` does an
in-place storage swap and requires *type-compatible* storages. CUDA tolerates the CPU→GPU swap path FSDP uses;
Habana doesn't. The general fix for non-CUDA + FSDP1 is "materialize on the device *before* wrapping."

### C6. `torch.hpu.empty_cache` doesn't exist on every build  →  `device.py` shim

verl calls `get_torch_device().empty_cache()` to release cached memory. Some Habana torch builds (2.7.1) don't have
`torch.hpu.empty_cache`. We shim a no-op when it's missing. **Concept:** the device-namespace duck-type isn't
100% — when you port through `getattr(torch, device)`, defensively shim the methods the framework assumes.

### C7. The rollout: the actual wall

This is where "verl on Gaudi" becomes "**LLM inference on Gaudi**," and it splits into two architectures:

#### C7a. vLLM rollout (`vllm`) — blocked by exclusive devices

The `vllm` rollout is the *good* path (fast, and verl 0.9's only supported rollout). vLLM-gaudi launches an
**`EngineCore` in a separate worker process**. That worker does `torch.ones(1).to('hpu')` and gets
`Device acquire failed` — because the FSDP **actor** process already holds the module. We got vLLM to *fully launch
and register all its HPU models*, but the colocation is fundamentally incompatible with Gaudi's per-process device
exclusivity. (Patches we *did* land here, in `env/`: `patch_strenum.py` shims Python-3.11 `enum.StrEnum` for the
3.10 image; an `empty_cache` shim; the fractional-HPU Ray patch.)

**The fix that remains:** *disaggregate* — give the FSDP actor and the vLLM server **different** Gaudi modules
(e.g. actor on HPU 0, vLLM on HPU 1) instead of colocating on one. verl supports separated placement; wiring it for
HPU is the recommended next step.

#### C7b. HF rollout (`hf`) + optimum-habana — got tokens out, then a lazy/eager catch-22

The `hf` rollout calls `model.generate()` **in the actor's own process** → no second device acquisition → it
*works around* the exclusivity problem. But it only exists in **verl 0.5.0** (0.9 deleted it), and vanilla HF
generate isn't HPU-compatible. The chain of fixes (all in `verl05.diff` + `env/patch_oh_*.py`), each a real concept:

1. **`adapt_transformers_to_gaudi()`** (optimum-habana) — monkeypatches `transformers` so models become
   `GaudiQwen2ForCausalLM` with a Gaudi-aware `generate()` (static shapes, `mark_step()` per token). Required a
   **matched pair** `transformers==4.49 + optimum-habana==1.18` (newer transformers broke its imports), plus
   shimming a few moved symbols.
2. **double-`n` bug** — verl's trainer already repeats the prompt batch by `n`; HFRollout *also* used
   `num_return_sequences=self.config.n` → `64*4=256` vs `64`. Fixed to `num_return_sequences=1`.
3. **FSDP wrap class** — after adapt, the layer class is `GaudiQwen2DecoderLayer`, but `_no_split_modules` still
   says `Qwen2DecoderLayer` → FSDP "Could not find the transformer layer class to wrap." Fixed by passing
   `wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]` (and note: it's iterated, so it must be a
   **list**, not a bare string — a subtle one).
4. **optimum-habana bug** — `_sample()` did `torch.tensor(eos_token_id)` on **CPU** then `torch.isin(hpu, cpu)`.
   We patched the library to put it on `input_ids.device`. (A genuine upstream bug.)
5. **The catch-22 (the final wall):**
   - `FSDP.summon_full_params()` (HFRollout gathers full weights to generate) frees shard storage via
     `storage._resize_(0)` → **"should not be called in lazy flow"** → needs **eager** mode.
   - But in **eager** mode, generation's view ops raise `synNodeCreateWithId failed: strided_view`, and
     optimum-habana's static cache clashes with bf16 autocast (`index_copy_ Float vs BFloat16`).
   - Skipping `summon_full_params` (to stay in lazy) crashed the worker outright.

   We removed the bf16 autocast (fixed the dtype clash), moved inputs to HPU, guarded the empty-slice argmax — and
   **generation ran for ~36 seconds on the Gaudi card** — but no single (lazy|eager) configuration satisfies *both*
   FSDP's storage ops *and* HF-generate's view ops at once. That is the honest end of this path.

**Concept (the big one):** *training* (FSDP) and *inference* (`generate`) make **opposite** demands on Habana's
execution mode. FSDP's storage-resize wants eager; optimum-habana's static-shape generate wants lazy. The
production answer is to **not run inference through the FSDP training graph at all** — i.e., use a dedicated
inference engine (vLLM-gaudi) on a *separate* device. Which loops us right back to C7a + disaggregation.

---

## D. The 30-second mental model of the whole port

```
  device.py / platform_hpu.py      ->  "hpu exists, talk hccl, see HABANA_VISIBLE_MODULES"
  ray base.py / ray_trainer.py     ->  "schedule on the HPU resource, count it, allow fractions"
  RAY_agent_register_timeout / etc.->  "let Ray actually boot on this cluster"
  attn flash->sdpa                 ->  "use a kernel that exists on HPU"
  FSDP pre-move (.to('hpu'))       ->  "don't make FSDP swap CPU<->HPU storage"
  empty_cache shim                 ->  "fill a gap in the device duck-type"
  rollout (vllm | hf+optimum)      ->  "do INFERENCE on Gaudi"  <-- the unsolved mile
```

Everything from `device.py` down to `empty_cache` is **"make the training half run on Gaudi"** — and it *does*.
The last line, the rollout, is **"make the inference half run on Gaudi *while colocated with training*"** — and
that is the wall, for the architectural reasons in C7.

Continue to [`03-debugging-journey.md`](03-debugging-journey.md) for the blow-by-blow (great for pattern-matching
future HPU errors), or [`memory.md`](memory.md) for exact current state + next steps.
