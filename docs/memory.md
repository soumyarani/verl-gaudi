# memory.md — Project State, Checkpoints, and Exact Next Steps

> If you are Claude (or a human) picking this up: read this top-to-bottom, then `skills.md` for the run recipes.
> Everything is checkpointed on ASU Sol at **`/scratch/ssamine4/verl_gaudi/`** (SSH host alias `sol`, needs ASU VPN).

Last updated: 2026-06-30.

## TL;DR status
- ✅ verl **installed and verified working on Intel Gaudi 2** (driver SynapseAI 1.24) — Ray + custom `PlatformHPU`
  backend + **FSDP model sharding on the Gaudi card** + optimizer + reward all run.
- ✅ Two working container environments built (PyTorch-1.24 and vLLM-1.24).
- ✅ **Generation ran ~36 s on the Gaudi card** via the `hf` rollout (verl 0.5.0 + optimum-habana).
- 🧱 **No full GRPO iteration completed.** The rollout/generation step hits HPU walls in *both* architectures.

## The two walls (and why)
1. **vLLM rollout (verl 0.9):** vLLM runs as a *separate process*; it can't acquire the Gaudi module the FSDP actor
   already holds (`Device acquire failed`, Gaudi modules are exclusive-per-process). vLLM itself launches fine on HPU.
2. **HF rollout (verl 0.5.0 + optimum-habana):** runs `model.generate()` through the FSDP graph → FSDP
   `summon_full_params` (`storage._resize_`) needs **eager** mode, but HF-generate view ops need **lazy** → a true
   catch-22. Both arrows point to the same fix:

## ➡️ Recommended next step: **disaggregated vLLM-gaudi**
Give the FSDP **actor** and the vLLM **rollout server** *different* Gaudi modules (e.g. actor on HPU 0, vLLM on
HPU 1) instead of colocating on one device. Then there is no device-acquire conflict and inference runs on the
purpose-built engine. Concretely to try:
- verl supports non-colocated / "standalone server" rollout placement. Investigate
  `actor_rollout_ref.rollout.nnodes` / replica placement and `n_gpus_per_node`/resource-pool split so the vLLM
  `ServerAdapter` lands on a *different* `HPU` resource bundle than the actor `WorkerDict`.
- Start from the working vLLM env (below). The vLLM engine already launches on HPU; only its colocation failed.
- Sanity-check first with a **standalone** `vllm_gaudi` serve of Qwen2.5-0.5B inside the vLLM container (no verl) to
  confirm generation works on its own HPU, then wire verl's disaggregated placement.

Alternative paths: (b) a Habana-validated `optimum-habana + transformers + verl` pin for the HF path; (c) upstream
fixes to optimum-habana `_sample` / FSDP-on-HPU storage ops.

## Cluster facts (Sol)
- Host alias `sol` = `login.sol.rc.asu.edu`, user `ssamine4`. **Needs ASU VPN** (VPN-only DNS; if it stops
  resolving the VPN dropped — reconnect; cannot be fixed agent-side).
- `gaudi` partition: 10 nodes, **8× HL-225 (Gaudi2, 92 GB)** each, driver **SynapseAI 1.24**. Rocky 8.10, SLURM.
- **Use `sbatch`, never interactive `srun`/`salloc`** (interactive job-steps fail on Sol).
- Run on a **gaudi node** (login node can't import HPU torch — old glibc). Compute nodes have internet.
- Some gaudi nodes occasionally have a stuck HPU (`Device acquire failed` even for the actor) — `--exclude` them.

## Workspace layout — `/scratch/ssamine4/verl_gaudi/`
| Path | What |
|------|------|
| `gaudi_124_pt210.sif` | Habana 1.24 **PyTorch** container (torch 2.10) — used for the HF-rollout / FSDP path |
| `gaudi_124_vllm.sif` | Habana 1.24 **vLLM** container (torch 2.7.1, **vllm 0.9.1 + vllm_gaudi 0.21**) — the disaggregation path |
| `verl/` | verl **0.9.0.dev0** clone (vLLM path) — patched; has `verl/plugin/platform/platform_hpu.py` |
| `verl05/` | verl **0.5.0** clone (HF-rollout path) — patched; **the path that generated tokens** |
| `cpkgs/` | writable user-site for the PyTorch container (`PYTHONUSERBASE`): verl deps, ray 2.55.1, **transformers 4.49 + optimum-habana 1.18** |
| `cpkgs_vllm/` | writable user-site for the vLLM container: verl deps, ray 2.47.1 (patched) |
| `models/Qwen2.5-0.5B-Instruct/` | the policy model |
| `data/gsm8k/{train,test}.parquet` | GSM8k preprocessed |
| `scripts/` | versioned run/setup scripts (`run_05.sh`, `_run_inside_05.sh`, `run_vllm.sh`, `setup_*`, ...) |
| `logs/` | all job logs (`run_05*.log` = HF path, `run_vllm*.log` = vLLM path) |
| `cconstraints.txt`, `cv_constraints.txt` | pip constraints pinning the HPU torch/numpy so deps can't clobber it |

## What's where in *this repo* (mirror of the changes)
- `patches/verl05.diff` — all verl-0.5.0 source changes (the running path).
- `patches/verl09_main.diff` — all verl-0.9 source changes (vLLM path; includes platform plugin registration).
- `patches/platform_hpu.py` — the new `PlatformHPU` backend (verl 0.9).
- `patches/env/patch_*.py` — patches to *installed libraries* (Ray `ray_option_utils`, optimum-habana `_sample`,
  `enum.StrEnum` shim) that live outside verl's git tree. Re-apply after reinstalling those libs.
- `scripts/` — the exact sbatch wrappers + inside-scripts to reproduce both paths.

## The two "magic" run commands (full detail in skills.md)
- HF path (generated tokens): `sbatch scripts/run_05.sh` → `_run_inside_05.sh` (verl05 + `rollout.name=hf` +
  optimum-habana, eager mode, `cpkgs`).
- vLLM path (engine launches): `sbatch scripts/run_vllm.sh` → `_run_inside_vllm.sh` (verl + `rollout.name=vllm` +
  `cpkgs_vllm`).

## ~20 patches, one-line each (ordered) — see `03-debugging-journey.md` for the *why*
1. `device.py`/`platform_hpu.py`: register `hpu` (device, `hccl`, `HABANA_VISIBLE_MODULES`).
2. `base.py`: HPU placement-group bundle + `{"resources":{"HPU":n}}` actor option.
3. `ray_trainer.py`: count `HPU` in `_check_resource_available`.
4. `main_ppo.py` (v0.5): `ray.init(resources={"HPU":8}, num_gpus=0, _node_ip_address=...)`.
5. Ray `ray_option_utils.py`: allow **fractional HPU**.
6. FSDP `EngineRegistry`: add `hpu` to FSDP engine (v0.9).
7. attn `flash_attention_2 -> sdpa` (3 sites).
8. FSDP **pre-move** `module.to('hpu')` before `FSDP(...)`.
9. `torch.hpu.empty_cache` no-op shim.
10. `enum.StrEnum` shim (py3.10) for vLLM image.
11. `RAY_agent_register_timeout_ms=300000` + explicit `HPU:8` resource (run-env).
12. `trainer.use_v1=False` (classic trainer, avoids `transfer_queue`).
13. HFRollout `num_return_sequences=1` (double-`n` fix).
14. `rollout.tensor_model_parallel_size=1`.
15. `adapt_transformers_to_gaudi()` wired into `device.py` (HF path) + `transformers==4.49`/`optimum-habana==1.18`.
16. `wrap_policy.transformer_layer_cls_to_wrap=[GaudiQwen2DecoderLayer]` (list!).
17. HFRollout: drop bf16 autocast; move inputs to HPU; pass `lazy_mode=True`.
18. HFRollout: skip `summon_full_params` on HPU (the lazy-mode attempt).
19. optimum-habana `_sample`: eos tensor `.to(input_ids.device)`.
20. optimum-habana `_sample`: guard empty eos-search slice; no-op `check_synapse_version`.

## Honest bottom line
> verl is installed and runs on Gaudi through the entire training pipeline up to and including generation — but the
> rollout/generation step hits HPU-specific op-compatibility walls in both available architectures (vLLM
> colocation; HF-generate+FSDP+optimum-habana). Completing a full iteration needs one of: **disaggregated vLLM**
> (dedicate separate HPUs to actor vs. rollout), a Habana-validated optimum-habana/transformers/verl pin for this
> exact path, or upstream fixes. All ~20 patches and both environments are checkpointed on
> `/scratch/ssamine4/verl_gaudi/`.
