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
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_json(args.data, lines=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts = []
    for p in df.prompt:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
    #prompts = sorted(prompts, key=lambda p: len(p["prompt"]))
    prompts = prompts[:150]

    model = LLM(
        model=args.model,
        dtype=torch.float16,
        tensor_parallel_size=1,
        max_model_len=32768
        #max_tokens=512
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=16000)

    outputs = model.generate(prompts, sampling_params)

    generation = [out.outputs[0].text for out in tqdm(outputs, desc="Generations", total=len(prompts))]

    df["generation"] = generation
    df.to_json(f"{args.output}/output.jsonl", orient='records', lines=True)

    df["prediction"] = df.generation.apply(lambda s: 1 if "\\boxed{proved}" in s.lower() else (0 if "\\boxed{disproved}" in s.lower() else -1))
    accs = [ (df[df.problem_name == p].answer == df[df.problem_name == p].prediction).all() for p in df.problem_name.unique() ]
    score = round(np.mean(accs) * 100, 4)
    print(f'Outcome score on fimo:', score)