f = "/scratch/ssamine4/verl_gaudi/cpkgs_vllm/lib/python3.10/site-packages/ray/_private/ray_option_utils.py"
s = open(f).read()
old = "    if resource_name in ray._private.accelerators.get_all_accelerator_resource_names():"
new = '    if resource_name != "HPU" and resource_name in ray._private.accelerators.get_all_accelerator_resource_names():'
if new in s:
    print("already patched")
else:
    assert old in s, "pattern not found"
    open(f,"w").write(s.replace(old,new)); print("patched ray 2.47.1 fractional HPU")
