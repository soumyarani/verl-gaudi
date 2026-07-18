# AIR-Gaudi disaggregated-GRPO patches

Source edits that make **disaggregated** (`separate_async`) GRPO run on Intel Gaudi 2 with `vllm_gaudi 0.24` + verl
`0.9.0.dev`. Full narrative: [`../../docs/AIR_GAUDI_DISAGG.md`](../../docs/AIR_GAUDI_DISAGG.md).

These are **idempotent** and are applied **inside the pod** against the *live installs* (paths default to the pod
layout: `/workspace/vllm/…`, `/workspace/vllm-gaudi/…`, `/workspace/verl-src/…`). Re-apply after any reinstall.

| Script | Layer | Target | What it does |
|---|---|---|---|
| `patch_air_capture_restore_base.py` | C | `hpu_model_runner.py` + `vllm_rollout/utils.py` | scaffold: capture param metadata after `get_model()`; restore-then-`load_weights` in `_update_weights` |
| `patch_air_weightsync_dict.py` | C | (same two files) | upgrade the capture/restore to **full `__dict__` + `__class__`** — the load-bearing fix for the "deadlock" |
| `patch_air_detok_neuter.py` | D1 | `vllm/tokenizers/detokenizer_utils.py` | skip cosmetic logprob detokenization (HPU emits out-of-range logprob token-ids) |
| `patch_air_logprob_tolerant.py` | D2 | `vllm_rollout/vllm_async_server.py` | default `0.0` when the sampled token is absent from the HPU logprobs dict |
| `patch_air_gsm8k_flexible.py` | reward | `verl/utils/reward_score/gsm8k.py` | GSM8k scorer -> flexible extraction (model answers in `\boxed{}`, strict `####` gave reward 0) |

Apply in order (C base → C dict → D1 → D2):
```bash
python patch_air_capture_restore_base.py
python patch_air_weightsync_dict.py
python patch_air_detok_neuter.py
python patch_air_logprob_tolerant.py
```

Also required but documented elsewhere (Layer A/B — Ray/K8s/checkpoint-engine):
- `runtimeClassName: habana` on the pod (device acquire) — see [`AIR_RUNBOOK.md`].
- Ray `max_colocate_count → 1` (`single_controller/ray/base.py`, `workers/rollout/replica.py`).
- CPU-only checkpoint engine `verl.plugin.checkpoint_engine.hccl_hpu` (`StatelessProcessGroup.create`, `ipc_collect`
  guard) — on the `vllm-gaudi` branch.
- `HpuModelAdapter.load_weights` delegation (in `fixes_final/hpu_model_runner.py`).

## `fixes_final/`
As-built full copies of the four patched files, for diffing/reference:
`hpu_model_runner.py`, `vllm_rollout_utils.py`, `vllm_async_server.py`, `detokenizer_utils.py`.
