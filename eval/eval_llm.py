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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--output", type=str, default="fimo")
    parser.add_argument("--timeout", type=int, default=120)
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

    def timeout_handler(signum, frame):
        raise TimeoutError

    generation = []
    for prompt in tqdm(prompts, desc="Generations"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)
        try:
            out = model.generate([prompt], sampling_params)
            gen_text = out[0].outputs[0].text
        except TimeoutError:
            gen_text = "TIMEOUT"
        except Exception as e:
            gen_text = f"ERROR: {str(e)}"
        finally:
            signal.alarm(0)  # disable alarm
        generation.append(gen_text)

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
    print(f'Outcome score on fimo:', score)
