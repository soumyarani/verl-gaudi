f = "/scratch/ssamine4/verl_gaudi/cpkgs/lib/python3.10/site-packages/ray/_common/ray_option_utils.py"
s = open(f).read()
old = "    if resource_name in ray._private.accelerators.get_all_accelerator_resource_names():"
new = '    if resource_name != "HPU" and resource_name in ray._private.accelerators.get_all_accelerator_resource_names():'
if new in s:
    print("ray already patched")
else:
    assert old in s, "pattern not found"
    open(f, "w").write(s.replace(old, new))
    print("patched ray _validate_resource_quantity to allow fractional HPU")
for l in open(f).read().splitlines():
    if 'resource_name != "HPU"' in l:
        print("NOW:", l.strip())
