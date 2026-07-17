import re

# ---- 1) hpu_model_runner: capture vLLM param metadata right after get_model() ----
hp = "/workspace/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py"
s = open(hp).read()
anchor1 = "                self.model = get_model(vllm_config=self.vllm_config)\n"
cap = anchor1 + (
    "                try:\n"
    "                    self._verl_saved_meta = {}\n"
    "                    for _vn, _vp in self.model.named_parameters():\n"
    "                        _vm = {_a: getattr(_vp, _a) for _a in ('weight_loader', 'weight_loader_v2',\n"
    "                               'output_dim', 'input_dim', 'ignore_warning', 'packed_dim', 'packed_factor',\n"
    "                               'shard_id') if hasattr(_vp, _a)}\n"
    "                        if _vm:\n"
    "                            self._verl_saved_meta[_vn] = _vm\n"
    "                except Exception:\n"
    "                    self._verl_saved_meta = {}\n"
)
assert anchor1 in s, "get_model anchor not found"
if "_verl_saved_meta = {}" not in s:
    s = s.replace(anchor1, cap, 1)
    open(hp, "w").write(s)
    print("hpu_model_runner: capture added")
else:
    print("hpu_model_runner: already has capture")

# ---- 2) utils.py: replace the messy load block with restore-then-load ----
up = "/workspace/verl-src/verl/workers/rollout/vllm_rollout/utils.py"
u = open(up).read()
old_block = (
    "                if param_updates:\n"
    "                    for model in self._iter_all_models():\n"
    "                        for _dn, _dp in model.named_parameters():\n"
    "                            if 'qkv' in _dn or 'q_proj' in _dn:\n"
    "                                logger.error('QKV_STATE %s wl=%s od=%s cls=%s' % (_dn, hasattr(_dp,'weight_loader'), getattr(_dp,'output_dim','NONE'), type(_dp).__name__)); break\n"
    "                        for _mod in model.modules():\n"
    "                            _wl = getattr(_mod, 'weight_loader', None)\n"
    "                            if callable(_wl):\n"
    "                                for _p in _mod.parameters(recurse=False):\n"
    "                                    if not hasattr(_p, 'weight_loader'):\n"
    "                                        _p.weight_loader = _wl\n"
    "                        try:\n"
    "                            model.load_weights(param_updates)\n"
    "                        except BaseException as _e:\n"
    "                            import traceback as _tb\n"
    "                            _names=[n for n,_ in param_updates][:6]\n"
    "                            logger.error(f\"LOAD_WEIGHTS_FAILED n={len(param_updates)} first={_names} err={type(_e).__name__}: {_e}\")\n"
    "                            _tb.print_exc()\n"
    "                            raise\n"
)
new_block = (
    "                if param_updates:\n"
    "                    _meta = getattr(getattr(self, 'model_runner', None), '_verl_saved_meta', None) or {}\n"
    "                    for model in self._iter_all_models():\n"
    "                        if _meta:\n"
    "                            for _n, _p in model.named_parameters():\n"
    "                                _sv = _meta.get(_n)\n"
    "                                if _sv is None:\n"
    "                                    for _sn, _sm in _meta.items():\n"
    "                                        if _n.endswith(_sn) or _sn.endswith(_n):\n"
    "                                            _sv = _sm\n"
    "                                            break\n"
    "                                if _sv:\n"
    "                                    for _a, _v in _sv.items():\n"
    "                                        if not hasattr(_p, _a):\n"
    "                                            setattr(_p, _a, _v)\n"
    "                        try:\n"
    "                            model.load_weights(param_updates)\n"
    "                        except BaseException as _e:\n"
    "                            import traceback as _tb\n"
    "                            logger.error('LOAD_WEIGHTS_FAILED n=%d err=%s: %s' % (len(param_updates), type(_e).__name__, _e))\n"
    "                            _tb.print_exc()\n"
    "                            raise\n"
)
assert old_block in u, "load block anchor not found"
u = u.replace(old_block, new_block, 1)
open(up, "w").write(u)
print("utils.py: restore-then-load installed")
