# AIR (ASU Gaudi 2 Kubernetes) runbook — persistent runs + reconnect

This is the **AIR Platform** path for the vLLM-on-Gaudi work (pivot from Sol's shared `gaudi` partition).
Everything heavy runs **inside a Kubernetes pod on the AIR cluster — server-side, NOT on the laptop** — so it keeps
running when the laptop is offline/asleep.

## ⭐ Persistence model (read this first)
- The compute lives in the **pod on the cluster**. It runs 24/7 until the pod is deleted/evicted — **independent of
  your laptop's internet/VPN.** Closing the laptop or losing wifi only pauses *monitoring*, never the run.
- **Always launch long jobs detached** so they survive an exec disconnect: either **`tmux`** inside the pod
  (preferred — re-attachable) or `nohup … &`.
  - A persistent tmux session named **`verl`** already exists in the pod (`tmux new-session -d -s verl -c /workspace`).
- **Caveat (storage):** the workspace is currently `emptyDir` (ephemeral) because the cluster's default **Longhorn**
  storage class fails to provision (`topologyKeys not found on any nodes` — a cluster-config bug on the Gaudi nodes).
  → the pod staying up = everything persists; **if the pod *restarts*, `/workspace` is lost** and the vLLM install must
  be redone. Ask RC to fix Longhorn topology (or give a working StorageClass) so we can mount a PVC for true
  cross-restart persistence of the env + model + checkpoints.

## Access
- kubeconfig: `~/.kube/config` (Rancher-managed cluster, `https://rancher.rc.asu.edu/k8s/clusters/local`).
- Namespace: **`user-ssamine4`** (namespaced service account — can create pods/jobs/deployments/PVCs/services/exec;
  cannot list nodes/storageclasses — that's normal).
- `kubectl` installed via `brew install kubectl`. Get access from **voyager.rc.asu.edu → Kubernetes tab** (download
  kubeconfig).

## Reconnect after being away (laptop back online)
```bash
# 1) connect ASU VPN, then:
export KUBECONFIG=~/.kube/config
kubectl get pods -n user-ssamine4                          # find the verl-dev pod
POD=$(kubectl get pods -n user-ssamine4 -l app=verl-dev -o jsonpath='{.items[0].metadata.name}')
# 2) re-attach the persistent session (interactive):
kubectl exec -it -n user-ssamine4 $POD -- tmux attach -t verl
# …or just tail a run's log without attaching:
kubectl exec -n user-ssamine4 $POD -- bash -lc 'tail -f /workspace/<run>.log'
```

## Environment (confirmed)
- Node: `dcx-gaudi0xx` (HL-225 / Gaudi2, 98 GB), **driver 1.24.0**, `hl-smi` OK.
- Base image: `vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest`
  (torch 2.11, `habana_frameworks.torch` + `hpu_available` work). No vLLM in the image — we install it.
- Pod requirements that matter: **`hostIPC: true`** (Habana shm — relevant to the layer-4 weight-sync);
  **cannot** add `SYS_NICE` capability (Kubewarden admission blocks it — omit it).
- Full internet egress (github/pypi/HF) → install at runtime.
- vLLM stack: **vllm-gaudi 0.24.0** (targets Gaudi 1.24.1) built per the official recipe
  (`VLLM_TARGET_DEVICE=empty`, reuses Habana torch). Install script at `/workspace/install_vllm.sh`.

## Status / next
1. ✅ Gaudi K8s access, clean 1.24 env, egress, tmux persistence.
2. 🔄 vLLM + vllm_gaudi 0.24 installed → **next: standalone vLLM generate smoke test on HPU.**
3. ⬜ Then the **layer-4 weight-sync test** — does this newer vLLM-Gaudi engine execute `update_weights_from_ipc`
   (HPU shared-memory weight load) that Sol's older stack never did? With `hostIPC` this is the best shot at closing
   the blocker documented in `VLLM_GAUDI.md`.
