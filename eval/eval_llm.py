if __name__ == "__main__":
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

    from transformers import AutoTokenizer
    import torch
    import pandas as pd
    import numpy as np
    from collections import Counter
    from vllm import LLM, SamplingParams
    from tqdm import tqdm
    import argparse
    import signal
    from contextlib import contextmanager

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_json(args.data, lines=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = []
    for p in df.prompt:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    model = LLM(
        model=args.model,
        dtype=torch.float16,
        tensor_parallel_size=2,
        max_model_len=32768,
        gpu_memory_utilization=0.90
    )

    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
        skip_special_tokens=True
    )

    @contextmanager
    def timeout(seconds):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Batch timed out after {seconds} seconds")
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    all_generations = []
    failed_indices = []
    
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + args.batch_size]
        
        try:
            with timeout(args.timeout):
                batch_outputs = model.generate(batch_prompts, sampling_params)
                batch_generations = [out.outputs[0].text for out in batch_outputs]
                all_generations.extend(batch_generations)
                print(f"Batch {i//args.batch_size + 1}: {len(batch_generations)} completed")
            
        except TimeoutError as e:
            print(f"TIMEOUT in batch {i//args.batch_size + 1}: {e}")
            all_generations.extend(["TIMEOUT"] * len(batch_prompts))
            failed_indices.extend(range(i, i + len(batch_prompts)))
            
        except Exception as e:
            print(f"ERROR in batch {i//args.batch_size + 1}: {e}")
            all_generations.extend(["ERROR"] * len(batch_prompts))
            failed_indices.extend(range(i, i + len(batch_prompts)))

    if failed_indices:
        print(f"WARNING: {len(failed_indices)} prompts failed: {failed_indices}")

    df["generation"] = all_generations
    df.to_json(f"{args.output}/output_intermediate.jsonl", orient='records', lines=True)
    print(f"Saved intermediate results to {args.output}/output_intermediate.jsonl")

    def extract_answer(s):
        s = s.lower()
        if "\\boxed{proved}" in s or "\\boxed{\\text{proved}}" in s:
            return 1
        if "\\boxed{disproved}" in s or "\\boxed{\\text{disproved}}" in s:
            return 0
        return -1

    df["prediction"] = df.generation.apply(extract_answer)
    
    accs = []
    for problem in df.problem_name.unique():
        df_problem = df[df.problem_name == problem]
        accs.append((df_problem.answer == df_problem.prediction).all())
    
    score = round(np.mean(accs) * 100, 4)
    df.to_json(f"{args.output}/output.jsonl", orient='records', lines=True)
    
    print(f'\n=== RESULTS ===')
    print(f'Answer distribution: {Counter(df.answer)}')
    print(f'Prediction distribution: {Counter(df.prediction)}')
    print(f'Outcome score on fimo: {score}%')
    print(f'Total prompts: {len(all_generations)}')
    print(f'Failed/Timeout: {len(failed_indices)}')
    print(f'Valid predictions: {len([p for p in df.prediction if p != -1])}')