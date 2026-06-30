f = "/scratch/ssamine4/verl_gaudi/cpkgs/lib/python3.10/site-packages/optimum/habana/utils.py"
s = open(f).read()
old = ('def check_synapse_version():\n'
       '    """\n'
       '    Checks whether the versions of SynapseAI and drivers have been validated for the current version of Optimum Habana.\n'
       '    """\n')
new = old + '    return  # patched: skip SynapseAI/driver version validation (HPU stack verified separately)\n'
if "skip SynapseAI/driver version validation" in s: print("already patched")
else:
    assert old in s, "pattern not found"
    open(f,"w").write(s.replace(old,new,1)); print("no-op'd check_synapse_version")
