#!/usr/bin/env python3
"""AIR-Gaudi fix 2a: neuter cosmetic logprob detokenization in vLLM.

vllm_gaudi's HPU top-k / logprobs op emits out-of-range / garbage token-ids
(e.g. -2117107376, near INT32_MIN) in the *logprob* token-id buffer. vLLM's
async output_handler then calls tokenizer.decode([token_id]) on them and the
Rust `tokenizers` binding raises `OverflowError: out of range integral type
conversion attempted`, which kills the EngineCore -> EngineGenerateError for
every request -> empty rollout batch -> `AssertionError: number of items:[0]`.

verl consumes only the NUMERIC logprobs, never the decoded logprob *strings*,
and the response text is produced by a SEPARATE incremental detokenizer. So we
skip the cosmetic per-logprob-token decode entirely. Idempotent.

Target: <vllm>/vllm/tokenizers/detokenizer_utils.py::convert_ids_list_to_tokens
Applied on the standalone vLLM install used by the rollout replicas (0.23.1rc1
pulled by vllm_gaudi 0.24).
"""
import sys

f = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vllm/vllm/tokenizers/detokenizer_utils.py"
s = open(f).read()

if 'skip cosmetic detok here' in s:
    print("detok already neutered")
    sys.exit(0)

start = s.index("    token_str_lst = []")
end = s.index("    return token_str_lst")
new = (
    "    token_str_lst = []\n"
    "    # HPU (vllm_gaudi) logprob top-k token-id buffer can contain out-of-range/garbage ids that\n"
    "    # crash tokenizer.decode; verl consumes only NUMERIC logprobs, so skip cosmetic detok here.\n"
    "    for token_id in token_ids:\n"
    "        token_str_lst.append(\"\")\n"
)
s = s[:start] + new + s[end:]
open(f, "w").write(s)
print("detok neutered (no decode)")
