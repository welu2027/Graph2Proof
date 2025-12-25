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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    parser.add_argument("--timeout", type=int, default=120)  # max seconds per prompt
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
        max_model_len=32768
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=16000)

    def generate_prompt(text):
        try:
            out = model.generate([text], sampling_params)
            return out[0].outputs[0].text
        except Exception:
            return "ERROR"

    generation = []
    total_prompts = len(prompts)
    checkpoint_idx = int(total_prompts * 0.94)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_idx = {executor.submit(generate_prompt, p): i for i, p in enumerate(prompts)}
        for future in tqdm(as_completed(future_to_idx), total=total_prompts, desc="Generations"):
            i = future_to_idx[future]
            try:
                gen_text = future.result(timeout=args.timeout)
            except TimeoutError:
                gen_text = "TIMEOUT"
            generation.append(gen_text)

            # Interim accuracy at ~94% processed
            if len(generation) == checkpoint_idx:
                df["generation"] = generation
                df["prediction"] = df.generation.apply(
                    lambda s: 1 if "\\boxed{proved}" in s.lower() else
                              (0 if "\\boxed{disproved}" in s.lower() else -1)
                )
                accs = [
                    (df[df.problem_name == p].answer == df[df.problem_name == p].prediction).all()
                    for p in df.problem_name.unique()
                ]
                score = round(np.mean(accs) * 100, 4)
                print(f"Checkpoint: 94% prompts processed. Interim outcome score: {score}%")

    # Final output
    df["generation"] = generation
    df.to_json(f"{args.output}/output.jsonl", orient='records', lines=True)

    df["prediction"] = df.generation.apply(
        lambda s: 1 if "\\boxed{proved}" in s.lower() else (0 if "\\boxed{disproved}" in s.lower() else -1)
    )
    accs = [
        (df[df.problem_name == p].answer == df[df.problem_name == p].prediction).all()
        for p in df.problem_name.unique()
    ]
    score = round(np.mean(accs) * 100, 4)
    print(f'Final outcome score on fimo:', score)
