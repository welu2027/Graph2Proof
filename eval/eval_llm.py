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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    parser.add_argument("--batch_size", type=int, default=8)  # NEW: Process in batches
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
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.90,  # NEW: More aggressive memory usage
        disable_log_stats=False  # NEW: Keep stats for debugging
    )

    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=16000,
        stop=["<|endoftext|>", "<|im_end|>"],  # NEW: Add stop tokens
        skip_special_tokens=True  # NEW: Clean output
    )

    # NEW: Process in batches with progress tracking
    all_generations = []
    failed_indices = []
    
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + args.batch_size]
        
        try:
            batch_outputs = model.generate(batch_prompts, sampling_params)
            batch_generations = [out.outputs[0].text for out in batch_outputs]
            all_generations.extend(batch_generations)
            
            # Print progress for slow batches
            print(f"Batch {i//args.batch_size + 1}: {len(batch_generations)} completed")
            
        except Exception as e:
            print(f"ERROR in batch {i//args.batch_size + 1}: {e}")
            # Fill with empty strings for failed batch
            all_generations.extend([""] * len(batch_prompts))
            failed_indices.extend(range(i, i + len(batch_prompts)))

    if failed_indices:
        print(f"WARNING: {len(failed_indices)} prompts failed: {failed_indices}")

    df["generation"] = all_generations
    
    # Save intermediate results
    df.to_json(f"{args.output}/output_intermediate.jsonl", orient='records', lines=True)
    print(f"Saved intermediate results to {args.output}/output_intermediate.jsonl")

    # Extract predictions
    def extract_answer(s):
        s = s.lower()
        if "\\boxed{proved}" in s or "\\boxed{\\text{proved}}" in s:
            return 1
        if "\\boxed{disproved}" in s or "\\boxed{\\text{disproved}}" in s:
            return 0
        return -1

    df["prediction"] = df.generation.apply(extract_answer)
    
    # Calculate accuracy
    accs = []
    for problem in df.problem_name.unique():
        df_problem = df[df.problem_name == problem]
        accs.append((df_problem.answer == df_problem.prediction).all())
    
    score = round(np.mean(accs) * 100, 4)
    
    # Final save
    df.to_json(f"{args.output}/output.jsonl", orient='records', lines=True)
    
    print(f'\n=== RESULTS ===')
    print(f'Answer distribution: {Counter(df.answer)}')
    print(f'Prediction distribution: {Counter(df.prediction)}')
    print(f'Outcome score on fimo: {score}%')
    print(f'Failed generations: {len([g for g in all_generations if g == ""])}')