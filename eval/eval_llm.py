if __name__ == "__main__":
    import os
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

    from transformers import AutoTokenizer
    import torch
    import pandas as pd
    import numpy as np
    from vllm import LLM, SamplingParams
    import argparse
    import json
    from openai import OpenAI
    from tqdm import tqdm

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

    PROCESS_PROMPT = '''You are an expert in scoring solutions for mathematical proof questions. The following question asks to prove or disprove a statement, where the statement may be either true or false. The test subject is asked to end their proof with \\boxed{proved} if they prove the statement to be true, and \\boxed{disproved} if they prove the statement to be false.

The question:

```
<question>
```

The ground truth of the statement:

```
<answer>
```

The test subject's solution:

```
<solution>
```

Your task is to evaluate the proof's quality and assign a score from 0 to 1 based on four criteria: logical validity (40%), completeness (30%), correctness (20%), and clarity (10%).

Instructions:

1. Analyze the proof step by step.
2. For each criterion:
   - Logical Validity: Check if each step follows logically from the previous one. Flag any logical errors.
   - Completeness: Verify if all necessary cases and steps are included to prove the theorem.
   - Correctness: Confirm if the final conclusion is correct.
   - Clarity: Assess if the proof is clear, unambiguous, and well-explained.
3. Assign a sub-score (0 to 1) for each criterion and compute the total score using the weights: (0.4 × validity) + (0.3 × completeness) + (0.2 × correctness) + (0.1 × clarity).
4. Provide a brief explanation (2-3 sentences) summarizing any errors or issues and justifying the score.

Final output format:

```
{
    "score": float,
    "validity": float,
    "completeness": float,
    "correctness": float,
    "clarity": float,
    "explanation": str
}
```

where "score" is the total score, and "validity", "completeness", "correctness", "clarity" are the subscores.'''

    def evaluate_process_gpt4o(client, prompt, answer, generation):
        """Call GPT-4o to evaluate proof quality"""
        eval_prompt = PROCESS_PROMPT.replace('<question>', prompt)
        eval_prompt = eval_prompt.replace('<answer>', "True" if answer else "False")
        eval_prompt = eval_prompt.replace('<solution>', generation)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0
            )
            
            result_text = response.choices[0].message.content
            # Extract JSON from response
            result_text = result_text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result
        except Exception as e:
            print(f"Error in process evaluation: {e}")
            return {
                "score": 0.0,
                "validity": 0.0,
                "completeness": 0.0,
                "correctness": 0.0,
                "clarity": 0.0,
                "explanation": f"Error: {str(e)}"
            }

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
    parser.add_argument("--process_eval", action="store_true", help="Run GPT-4o process evaluation")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key for process eval")
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

            print(f"[{len(gen_texts)}/{total_prompts}] Interim outcome score: {round(np.mean(accs)*100, 4)}%")

    # ---------- final results ----------
    generation_sorted = sorted(generation, key=lambda x: x[0])
    df["generation"] = [g[1] for g in generation_sorted]
    df["prediction"] = df.generation.apply(extract_prediction)

    # ---------- process evaluation ----------
    if args.process_eval:
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: --openai_api_key required or set OPENAI_API_KEY env var")
        else:
            print("\n" + "="*80)
            print("Running GPT-4o Process Evaluation...")
            print("="*80)
            
            client = OpenAI(api_key=api_key)
            process_scores = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Process eval"):
                result = evaluate_process_gpt4o(
                    client,
                    row['prompt'],
                    row['answer'],
                    row['generation']
                )
                process_scores.append(result)
            
            df['process_score'] = [r['score'] for r in process_scores]
            df['validity'] = [r['validity'] for r in process_scores]
            df['completeness'] = [r['completeness'] for r in process_scores]
            df['correctness'] = [r['correctness'] for r in process_scores]
            df['clarity'] = [r['clarity'] for r in process_scores]
            df['explanation'] = [r['explanation'] for r in process_scores]
            
            avg_process = df['process_score'].mean()
            print(f"\nAverage Process Score: {round(avg_process * 100, 4)}%")

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
    print(f"Outcome Score: {score}%")
    print(f"Variant-groups solved: {sum(accs)}/{len(accs)}")
    
    if args.process_eval and 'process_score' in df.columns:
        avg_process = df['process_score'].mean()
        avg_validity = df['validity'].mean()
        avg_completeness = df['completeness'].mean()
        avg_correctness = df['correctness'].mean()
        avg_clarity = df['clarity'].mean()
        
        print(f"\nProcess Score: {round(avg_process * 100, 4)}%")
        print(f"  - Validity: {round(avg_validity * 100, 4)}%")
        print(f"  - Completeness: {round(avg_completeness * 100, 4)}%")
        print(f"  - Correctness: {round(avg_correctness * 100, 4)}%")
        print(f"  - Clarity: {round(avg_clarity * 100, 4)}%")
    
    print("="*80)