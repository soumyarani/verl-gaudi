# Capture FULL __dict__ (all instance state: tp_rank, tp_size, output_dim, weight_loader, ...) + class
hp="/workspace/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py"
s=open(hp).read()
old=("                        _vm = {_a: getattr(_vp, _a) for _a in ('weight_loader', 'weight_loader_v2',\n"
     "                               'output_dim', 'input_dim', 'ignore_warning', 'packed_dim', 'packed_factor',\n"
     "                               'shard_id') if hasattr(_vp, _a)}\n"
     "                        _vm['_cls'] = type(_vp)\n")
new=("                        _vm = dict(getattr(_vp, '__dict__', {}) or {})\n"
     "                        _vm['_cls'] = type(_vp)\n")
assert old in s, "cap anchor missing"
open(hp,"w").write(s.replace(old,new,1)); print("capture full __dict__")

# Restore: set __class__ then __dict__.update (full instance state)
up="/workspace/verl-src/verl/workers/rollout/vllm_rollout/utils.py"
u=open(up).read()
oldr=("                                    for _a, _v in _sv.items():\n"
      "                                        if _a != '_cls' and not hasattr(_p, _a):\n"
      "                                            try:\n"
      "                                                setattr(_p, _a, _v)\n"
      "                                            except Exception:\n"
      "                                                pass\n")
newr=("                                    try:\n"
      "                                        _p.__dict__.update({_k: _vv for _k, _vv in _sv.items() if _k != '_cls'})\n"
      "                                    except Exception:\n"
      "                                        pass\n")
assert oldr in u, "restore anchor missing"
open(up,"w").write(u.replace(oldr,newr,1)); print("restore full __dict__")
