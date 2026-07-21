#!/usr/bin/env python3
"""Route data_source='livecodebench' to the SDPO code reward in verl's default_compute_score.
The separate_async reward-loop plumbing doesn't reliably pick up custom_reward_function.path,
so we register the route directly. Target: verl/utils/reward_score/__init__.py"""
import sys
f = sys.argv[1] if len(sys.argv)>1 else "/workspace/verl-src/verl/utils/reward_score/__init__.py"
s = open(f).read()
anchor = '    if data_source == "openai/gsm8k":'
inject = ('    if str(data_source) == "livecodebench":\n'
          '        import sys\n'
          '        if "/workspace" not in sys.path: sys.path.insert(0, "/workspace")\n'
          '        from lcb_reward import compute_score as _lcb\n'
          '        return _lcb(str(data_source), solution_str, ground_truth, extra_info)\n' + anchor)
if "_lcb" in s: print("already routed")
else:
    assert anchor in s; open(f,"w").write(s.replace(anchor, inject, 1)); print("livecodebench route added")
