f = "/scratch/ssamine4/verl_gaudi/verl/verl/plugin/platform/platform_hpu.py"
s = open(f).read()
anchor = "import torch\n"
shim = ("import enum as _enum\n"
        "if not hasattr(_enum, \"StrEnum\"):\n"
        "    class _StrEnum(str, _enum.Enum):\n"
        "        def __str__(self):\n"
        "            return str(self.value)\n"
        "    _enum.StrEnum = _StrEnum\n"
        "import torch\n")
if "_enum.StrEnum = _StrEnum" in s:
    print("already patched")
else:
    assert anchor in s, "anchor not found"
    # replace only the first occurrence of 'import torch\n'
    s = s.replace(anchor, shim, 1)
    open(f, "w").write(s); print("added enum.StrEnum shim to platform_hpu.py")
