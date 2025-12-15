if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch
    import pandas as pd
    import numpy as np
    from collections import Counter
    from vllm import LLM, SamplingParams
    from tqdm import tqdm
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    args = parser.parse_args()

    model_path = args.model
    data_file = args.data

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_json(data_file, lines=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt_template = '''<question>

Use natural language and LaTeX in the proof. If the statement is proved, add \\boxed{proved} at the end of your answer. If it's disproved, add \\boxed{disproved} at the end of your answer.'''

    prompts = []
    for p in df.prompt:
        messages = [
            {"role": "user", "content": prompt_template.replace("<question>", p)}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        prompts.append(text)
    print(prompts[0])
    print('---' * 10)

    model = LLM(
        model=model_path,
        dtype=torch.float16,  # Use float16 to reduce memory footprint
        tensor_parallel_size=1  # Single GPU to avoid MP issues and OOM
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=32000)
    print("Starting generation (vLLM single-GPU mode)...")
    outputs = model.generate(prompts, sampling_params)

    generation = []
    stop_reasons = []
    for output in tqdm(outputs, desc="Processing generations", total=len(prompts)):
        generation.append(output.outputs[0].text)
        stop_reasons.append(output.outputs[0].finish_reason)

    print(generation[0])
    print('---' * 10)
    print(Counter(stop_reasons))

    df["generation"] = generation
    df.to_json(f"{args.output}/{os.path.basename(model_path)}.jsonl", orient='records', lines=True)

    def extract_answer(s):
        s = s.lower()
        if "\\boxed{proved}" in s or "\\boxed{\\text{proved}}" in s:
            return 1
        if "\\boxed{disproved}" in s or "\\boxed{\\text{disproved}}" in s:
            return 0
        return -1

    df["prediction"] = df.generation.apply(extract_answer)
    print('Answer distribution:', Counter(df.answer))
    print('Prediction distribution:', Counter(df.prediction))

    accs = []
    for problem in set(df.problem_name):
        df_ = df[df.problem_name == problem]
        accs.append((df_.answer == df_.prediction).prod())

    score = round(np.mean(accs) * 100, 4)
    print(f'Outcome score on {os.path.basename(args.data).split(".")[0]}:', score)