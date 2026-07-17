f="/workspace/verl-src/verl/workers/rollout/vllm_rollout/vllm_async_server.py"
s=open(f).read()
old="            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]"
new=("            # HPU (vllm_gaudi) logprobs dict does not always contain the sampled token id;\n"
     "            # rollout log_probs are unused for the GRPO loss (actor recomputes old_log_probs), so default 0.0.\n"
     "            log_probs = [(logprobs[token_ids[i]].logprob if (logprobs is not None and token_ids[i] in logprobs) else 0.0)\n"
     "                         for i, logprobs in enumerate(final_res.outputs[0].logprobs)]")
assert old in s, "logprob line anchor missing"
open(f,"w").write(s.replace(old,new,1)); print("logprob tolerant patch applied")
