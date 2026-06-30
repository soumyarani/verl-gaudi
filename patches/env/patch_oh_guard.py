f = "/scratch/ssamine4/verl_gaudi/cpkgs/lib/python3.10/site-packages/optimum/habana/transformers/generation/utils.py"
s = open(f).read()
old = "        if batch_size > 1 and has_eos_stopping_criteria:\n            eos_token_id = generation_config.eos_token_id"
new = "        if batch_size > 1 and has_eos_stopping_criteria and input_ids[:, start_token_idx:].shape[1] > 0:\n            eos_token_id = generation_config.eos_token_id"
if "input_ids[:, start_token_idx:].shape[1] > 0" in s: print("already guarded")
else:
    assert old in s, "pattern not found"
    open(f,"w").write(s.replace(old,new,1)); print("guarded empty-slice eos masking in optimum-habana _sample")
