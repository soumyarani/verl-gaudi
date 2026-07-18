#!/usr/bin/env python3
"""AIR-Gaudi fix 3: GSM8k reward -> flexible answer extraction.

Qwen2.5-0.5B-Instruct SOLVES GSM8k but answers in its native `\boxed{72}` format,
while verl's gsm8k reward defaults to method="strict" which requires the literal
`#### 72`. Result: reward = 0 for every sample -> advantage 0 -> pg_loss 0 (no
learning), even though the model is correct. This is NOT a truncation issue
(raising max_response_length did not help).

FIX: default the gsm8k scorer to "flexible" (extracts the last number in the
response), which correctly scores \boxed{72} -> 1.0 and a wrong \boxed{99} -> 0.0.

Verified: default_compute_score("openai/gsm8k", r"...\boxed{72}", "72") == 1.0.
After this, a real GRPO step logged critic/rewards/mean=0.14, actor/pg_loss=0.0075.

Target: <verl>/verl/utils/reward_score/gsm8k.py::compute_score
"""
import sys
f = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl-src/verl/utils/reward_score/gsm8k.py"
s = open(f).read()
old = 'def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):'
new = 'def compute_score(solution_str, ground_truth, method="flexible", format_score=0.0, score=1.0):'
if new in s:
    print("gsm8k already flexible")
elif old in s:
    open(f, "w").write(s.replace(old, new, 1))
    print("gsm8k reward -> flexible default")
else:
    sys.exit("anchor not found: compute_score signature")
