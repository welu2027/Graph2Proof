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
    parser.add_argument("--limit", type=int, default=None, help="Only process first N problems")
    parser.add_argument("--print_proofs", action="store_true", help="Print generated proofs to console")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_json(args.data, lines=True)
    
    # Apply limit if specified
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"Limiting to first {args.limit} problems")
    
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

    def generate_prompt(text):
        try:
            out = model.generate([text], sampling_params)
            return out[0].outputs[0].text
        except Exception:
            return "ERROR"

    generation = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {
            executor.submit(generate_prompt, p): i
            for i, p in enumerate(prompts)
        }

        for future in tqdm(
            as_completed(future_to_idx),
            total=total_prompts,
            desc="Generations"
        ):
            idx = future_to_idx[future]
            try:
                gen_text = future.result(timeout=args.timeout)
            except TimeoutError:
                gen_text = "TIMEOUT"

            generation.append((idx, gen_text))
            processed = len(generation)

            # Print proof if flag is set
            if args.print_proofs or args.limit == 1:
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

                accs = [
                    (
                        df_partial[df_partial.problem_name == p].answer
                        == df_partial[df_partial.problem_name == p].prediction
                    ).all()
                    for p in df_partial.problem_name.unique()
                ]

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

    accs = [
        (
            df[df.problem_name == p].answer
            == df[df.problem_name == p].prediction
        ).all()
        for p in df.problem_name.unique()
    ]

    score = round(np.mean(accs) * 100, 4)
    print(f"\nFinal outcome score on fimo: {score}%")