if __name__ == "__main__":
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

    from transformers import AutoTokenizer
    import torch
    import pandas as pd
    import numpy as np
    from vllm import LLM, SamplingParams
    import argparse

    # ---------- helpers ----------
    def chunk_indices(indices, chunk_size=2):
        for i in range(0, len(indices), chunk_size):
            yield indices[i:i + chunk_size]

    def extract_prediction(text):
        t = text.lower()
        if any(x in t for x in [
            "\\boxed{proved}", "\\boxed{\\text{proved}}",
            "\\boxed{true}", "\\boxed{\\text{true}}"
        ]):
            return 1
        if any(x in t for x in [
            "\\boxed{disproved}", "\\boxed{\\text{disproved}}",
            "\\boxed{false}", "\\boxed{\\text{false}}"
        ]):
            return 0
        return -1

    # ---------- args ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--print_batch_size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_problems", type=int, default=None)
    parser.add_argument("--problem_name", type=str, default=None)
    parser.add_argument("--print_proofs", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---------- load data ----------
    df = pd.read_json(args.data, lines=True)

    if args.problem_name is not None:
        df = df[df.problem_name == args.problem_name].reset_index(drop=True)
        print(f"Processing problem '{args.problem_name}' with {len(df)} variants")
    elif args.limit_problems is not None:
        problems = df.problem_name.unique()[:args.limit_problems]
        df = df[df.problem_name.isin(problems)].reset_index(drop=True)
        print(f"Processing first {args.limit_problems} problems ({len(df)} variants)")
    elif args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)

    print(f"Total rows: {len(df)}")
    print(f"Unique problems: {df.problem_name.nunique()}")

    # ---------- tokenizer ----------
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

    # ---------- model ----------
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

    # ---------- generation ----------
    generation = []
    total_prompts = len(prompts)

    for batch_start in range(0, total_prompts, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_prompts)
        batch_prompts = prompts[batch_start:batch_end]

        print(f"\nGenerating [{batch_start+1}-{batch_end}] / {total_prompts}")
        outputs = model.generate(batch_prompts, sampling_params)

        for i, output in enumerate(outputs):
            idx = batch_start + i
            gen_text = output.outputs[0].text
            generation.append((idx, gen_text))

            if args.print_proofs:
                print("="*80)
                print(f"{df.iloc[idx]['problem_name']}")
                print(gen_text)
                print("="*80)

        # ---------- interim scoring ----------
        if batch_end % args.print_batch_size == 0 or batch_end == total_prompts:
            gen_sorted = sorted(generation, key=lambda x: x[0])
            gen_texts = [g[1] for g in gen_sorted]

            df_partial = df.iloc[:len(gen_texts)].copy()
            df_partial["generation"] = gen_texts
            df_partial["prediction"] = df_partial.generation.apply(extract_prediction)

            accs = []
            for p in df_partial.problem_name.unique():
                df_prob = df_partial[df_partial.problem_name == p]
                idxs = df_prob.index.tolist()

                for chunk in chunk_indices(idxs, 2):
                    df_chunk = df_prob.loc[chunk]
                    accs.append((df_chunk.answer == df_chunk.prediction).all())

            print(f"[{len(gen_texts)}/{total_prompts}] Interim score: {round(np.mean(accs)*100, 4)}%")

    # ---------- final results ----------
    generation_sorted = sorted(generation, key=lambda x: x[0])
    df["generation"] = [g[1] for g in generation_sorted]
    df["prediction"] = df.generation.apply(extract_prediction)

    df.to_json(
        f"{args.output}/output.jsonl",
        orient="records",
        lines=True
    )

    accs = []
    for p in df.problem_name.unique():
        df_prob = df[df.problem_name == p]
        idxs = df_prob.index.tolist()

        for chunk in chunk_indices(idxs, 2):
            df_chunk = df_prob.loc[chunk]
            accs.append((df_chunk.answer == df_chunk.prediction).all())

    score = round(np.mean(accs) * 100, 4)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final outcome score: {score}%")
    print(f"Variant-groups solved: {sum(accs)}/{len(accs)}")
    print("="*80)
