#!/usr/bin/env python3
"""LiveCodeBench v6 -> verl parquet, MATCHING SDPO (github.com/lasgroup/SDPO data/utils/livecodebench.py):
date split (train<2025-02-01, test>=2025-02-01), CODE_PROMPT, ground_truth={inputs,outputs,testtype,fn_name,time_limit},
data_source='livecodebench' -> routes to SDPO feedback/code.py reward. Caps tests for reward speed on HPU."""
import json, base64, zlib, pickle, os, random
from datetime import datetime
import pandas as pd
from huggingface_hub import hf_hub_download

CUTOFF = datetime(2025, 2, 1)
TIME_LIMIT = 6
MAX_TESTS = 15          # HPU reward-speed cap (SDPO uses all; note the deviation)
CODE_PROMPT = ("You are a coding expert. You will be given a coding problem, and you need to write a correct "
    "Python program that matches the specification and passes all tests. The time limit is 1 second. You may "
    "start by outlining your thought process. In the end, please provide the complete code in a code block "
    "enclosed with ``` ```.\n\n{problem}")
random.seed(0)

def parse_sig(starter):
    return "def " + starter.split("def ")[1].split("Input\n")[0].strip()

def translate_tests(encoded, fn_name):
    tests = json.loads(pickle.loads(zlib.decompress(base64.b64decode(encoded))))
    if len(tests) > MAX_TESTS:
        tests = tests[:1] + random.sample(tests[1:], MAX_TESTS-1)
    return {"inputs":[t["input"] for t in tests], "outputs":[t["output"] for t in tests],
            "testtype": tests[0]["testtype"], "fn_name": fn_name, "time_limit": TIME_LIMIT}

def build(ex):
    if not ex.get("private_test_cases"): return None
    problem = ex["question_content"]
    if ex.get("starter_code","").strip():
        problem += f"\n\nYour solution should have the following signature: ```python\n{parse_sig(ex['starter_code'])}\n```"
    fn_name = json.loads(ex["metadata"]).get("func_name","") if ex.get("metadata","").strip() else ""
    gt = translate_tests(ex["private_test_cases"], fn_name)
    if not gt["inputs"]: return None
    return {"data_source":"livecodebench",
            "prompt":[{"role":"user","content":CODE_PROMPT.format(problem=problem)}],
            "ability":"code",
            "reward_model":{"style":"rule","ground_truth":json.dumps(gt, ensure_ascii=False)},
            "extra_info":{"question_id":ex.get("question_id",""),"difficulty":ex.get("difficulty",""),
                          "contest_date":ex.get("contest_date",""),"testtype":gt["testtype"],"n_tests":len(gt["inputs"])}}

p = hf_hub_download("livecodebench/code_generation_lite","test6.jsonl",repo_type="dataset",revision="refs/pr/6")
rows=[json.loads(l) for l in open(p)]
print("v6 problems:", len(rows))
train,test=[],[]
for ex in rows:
    b=build(ex)
    if not b: continue
    d=datetime.fromisoformat(ex["contest_date"])
    b["extra_info"]["split"]="test" if d>=CUTOFF else "train"
    (test if d>=CUTOFF else train).append(b)
os.makedirs("/workspace/data/lcb_v6",exist_ok=True)
pd.DataFrame(train).to_parquet("/workspace/data/lcb_v6/train.parquet")
pd.DataFrame(test).to_parquet("/workspace/data/lcb_v6/test.parquet")
print(f"train(<2025-02)={len(train)} test(>=2025-02)={len(test)} -> /workspace/data/lcb_v6/")
print("testtype dist:", pd.Series([r['extra_info']['testtype'] for r in train]).value_counts().to_dict())
