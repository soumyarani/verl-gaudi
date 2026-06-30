f = "/scratch/ssamine4/verl_gaudi/cpkgs/lib/python3.10/site-packages/optimum/habana/transformers/generation/utils.py"
s = open(f).read()
old = "torch.isin(input_ids[:, start_token_idx:], torch.tensor(eos_token_id))"
new = "torch.isin(input_ids[:, start_token_idx:], torch.tensor(eos_token_id, device=input_ids.device))"
n = s.count("torch.tensor(eos_token_id)")
if "torch.tensor(eos_token_id, device=" in s: print("already patched")
else:
    assert old in s, "pattern not found"
    # fix all bare torch.tensor(eos_token_id) occurrences to be on input_ids.device
    s = s.replace(old, new)
    open(f,"w").write(s); print(f"patched optimum-habana _sample eos tensor device (this occurrence); total bare occ was {n}")
# show remaining bare ones
for i,l in enumerate(open(f).read().splitlines(),1):
    if "torch.tensor(eos_token_id)" in l: print(f"REMAINING bare L{i}: {l.strip()[:90]}")
