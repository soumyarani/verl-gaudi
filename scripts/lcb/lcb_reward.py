# verl custom_reward_function for LiveCodeBench (SDPO feedback/code.py, standalone).
import sys
sys.path.insert(0, "/workspace")
import lcb_code_reward as _c
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kw):
    ei = extra_info or {"split": "train"}
    if "split" not in ei: ei = {**ei, "split": "train"}
    res = _c.compute_score(solution_str, ground_truth, ei, sparse_rewards=True, max_test_cases=15)
    return res   # dict with score + feedback; verl uses res["score"]
