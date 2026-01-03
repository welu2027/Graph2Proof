if __name__ == "__main__":
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

    from transformers import AutoTokenizer
    import torch
    import pandas as pd
    import numpy as np
    from vllm import LLM, SamplingParams
    from tqdm import tqdm
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=10, help="Print accuracy every N generations")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    parser.add_argument("--limit_problems", type=int, default=None, help="Only process first N unique problems (all their variants)")
    parser.add_argument("--problem_name", type=str, default=None, help="Only process specific problem by name (all its variants)")
    parser.add_argument("--print_proofs", action="store_true", help="Print generated proofs to console")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_json(args.data, lines=True)
    
    # Apply filtering based on arguments
    if args.problem_name is not None:
        df = df[df.problem_name == args.problem_name].reset_index(drop=True)
        if len(df) == 0:
            print(f"ERROR: No problem found with name '{args.problem_name}'")
            print(f"Available problems: {sorted(pd.read_json(args.data, lines=True).problem_name.unique())}")
            exit(1)
        print(f"Processing problem '{args.problem_name}' with {len(df)} variants")
    elif args.limit_problems is not None:
        unique_problems = df.problem_name.unique()[:args.limit_problems]
        df = df[df.problem_name.isin(unique_problems)].reset_index(drop=True)
        print(f"Processing first {args.limit_problems} problems ({len(df)} total variants)")
    elif args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"Processing first {args.limit} rows")
    
    print(f"Total rows to process: {len(df)}")
    print(f"Unique problems: {df.problem_name.nunique()}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = []
    for p in df.prompt:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(text)

    total_prompts = len(prompts)
    print(f"Total prompts to process: {total_prompts}")

    model = LLM(
        model=args.model,
        dtype=torch.float16,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=32768
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=16000
    )

    # Generate all prompts at once (vLLM handles batching efficiently)
    print("Generating proofs...")
    outputs = model.generate(prompts, sampling_params)
    
    generation = []
    for idx, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        gen_text = output.outputs[0].text
        generation.append((idx, gen_text))
        processed = len(generation)

        # Print proof if flag is set or if testing single problem/small dataset
        if args.print_proofs or args.problem_name or args.limit_problems == 1 or (args.limit and args.limit <= 5):
            print("\n" + "="*80)
            print(f"PROBLEM #{idx + 1}: {df.iloc[idx]['problem_name']}")
            print("="*80)
            print("PROMPT:")
            print(df.iloc[idx]['prompt'][:500] + "..." if len(df.iloc[idx]['prompt']) > 500 else df.iloc[idx]['prompt'])
            print("\n" + "-"*80)
            print("GENERATED PROOF:")
            print(gen_text)
            print("-"*80)
            
            # Extract prediction - more flexible matching
            gen_lower = gen_text.lower()
            if "\\boxed{proved}" in gen_lower or "\\boxed{\\text{proved}}" in gen_lower or "\\boxed{true}" in gen_lower or "\\boxed{\\text{true}}" in gen_lower:
                prediction = 1
            elif "\\boxed{disproved}" in gen_lower or "\\boxed{\\text{disproved}}" in gen_lower or "\\boxed{false}" in gen_lower or "\\boxed{\\text{false}}" in gen_lower:
                prediction = 0
            else:
                prediction = -1
            
            answer = df.iloc[idx]['answer']
            correct = prediction == answer
            
            print(f"CORRECT ANSWER: {'proved' if answer == 1 else 'disproved'}")
            print(f"MODEL PREDICTION: {'proved' if prediction == 1 else ('disproved' if prediction == 0 else 'NONE')}")
            print(f"RESULT: {'✓ CORRECT' if correct else '✗ INCORRECT'}")
            print("="*80 + "\n")

        if processed % args.batch_size == 0 or processed == total_prompts:
                # Sort by index to maintain order
                generation_sorted = sorted(generation, key=lambda x: x[0])
                gen_texts = [g[1] for g in generation_sorted]
                
                df_partial = df.iloc[:processed].copy()
                df_partial["generation"] = gen_texts

                df_partial["prediction"] = df_partial.generation.apply(
                    lambda s: 1 if any(x in s.lower() for x in ["\\boxed{proved}", "\\boxed{\\text{proved}}", "\\boxed{true}", "\\boxed{\\text{true}}"])
                    else (0 if any(x in s.lower() for x in ["\\boxed{disproved}", "\\boxed{\\text{disproved}}", "\\boxed{false}", "\\boxed{\\text{false}}"]) else -1)
                )

                accs = []
                for p in df_partial.problem_name.unique():
                    df_prob = df_partial[df_partial.problem_name == p]
                    all_correct = (df_prob.answer == df_prob.prediction).all()
                    accs.append(all_correct)
                    
                    # If testing specific problem, show per-problem results
                    if args.problem_name:
                        variants_correct = (df_prob.answer == df_prob.prediction).sum()
                        total_variants = len(df_prob)
                        status = "✓ ALL VARIANTS CORRECT" if all_correct else f"✗ {variants_correct}/{total_variants} variants correct"
                        print(f"\nProblem '{p}': {status}")

                score = round(np.mean(accs) * 100, 4)
                print(f"[{processed}/{total_prompts}] Interim outcome score: {score}%")
    # Sort final generation by index
    generation_sorted = sorted(generation, key=lambda x: x[0])
    gen_texts = [g[1] for g in generation_sorted]
    
    df["generation"] = gen_texts
    df.to_json(
        f"{args.output}/output.jsonl",
        orient="records",
        lines=True
    )

    df["prediction"] = df.generation.apply(
        lambda s: 1 if any(x in s.lower() for x in ["\\boxed{proved}", "\\boxed{\\text{proved}}", "\\boxed{true}", "\\boxed{\\text{true}}"])
        else (0 if any(x in s.lower() for x in ["\\boxed{disproved}", "\\boxed{\\text{disproved}}", "\\boxed{false}", "\\boxed{\\text{false}}"]) else -1)
    )

    accs = []
    for p in df.problem_name.unique():
        df_prob = df[df.problem_name == p]
        all_correct = (df_prob.answer == df_prob.prediction).all()
        accs.append(all_correct)

    score = round(np.mean(accs) * 100, 4)
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    
    # Show detailed breakdown for specific problem or small problem sets
    if args.problem_name or (args.limit_problems and args.limit_problems <= 5):
        for p in df.problem_name.unique():
            df_prob = df[df.problem_name == p]
            all_correct = (df_prob.answer == df_prob.prediction).all()
            variants_correct = (df_prob.answer == df_prob.prediction).sum()
            total_variants = len(df_prob)
            status = "✓ ALL VARIANTS CORRECT" if all_correct else f"✗ ONLY {variants_correct}/{total_variants} CORRECT"
            print(f"Problem '{p}': {status}")
        print(f"{'='*80}")
    
    print(f"Final outcome score: {score}%")
    print(f"Problems solved: {sum(accs)}/{len(accs)}")
    print(f"{'='*80}")