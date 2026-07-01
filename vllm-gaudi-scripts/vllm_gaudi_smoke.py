import os, time

def main():
    print("=== vLLM-Gaudi standalone smoke test ===", flush=True)
    import vllm
    print("vllm", vllm.__version__, flush=True)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(
        model="/scratch/ssamine4/verl_gaudi/models/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1, dtype="bfloat16", enforce_eager=True,
        gpu_memory_utilization=0.4, max_model_len=1024,
    )
    print(f"LLM init: {time.time()-t0:.1f}s", flush=True)
    prompts = ["Question: What is 2+2?\nAnswer:", "The capital of France is"]
    t1 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=32, temperature=0.0))
    print(f"generate: {time.time()-t1:.2f}s for {len(prompts)} prompts", flush=True)
    for p, o in zip(prompts, outs):
        print(f"PROMPT: {p!r}\n  -> {o.outputs[0].text!r}", flush=True)
    print("VLLM_GAUDI_SMOKE_OK", flush=True)

if __name__ == "__main__":
    main()
