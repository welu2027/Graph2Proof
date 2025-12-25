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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
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
        max_model_len=32768
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=16000)

    generation = []
    total_prompts = len(prompts)
    checkpoint_idx = int(total_prompts * 0.93)

    outputs = model.generate(prompts, sampling_params)
    for i, out in enumerate(tqdm(outputs, desc="Generations", total=total_prompts)):
        generation.append(out.outputs[0].text)

        if i + 1 == checkpoint_idx:
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
