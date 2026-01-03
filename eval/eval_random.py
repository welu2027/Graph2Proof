import argparse
import pandas as pd
import numpy as np
import random

# ---------- helpers ----------
def chunk_indices(indices):
    """Randomly group variants in 1 or 2 (like your variant grouping)."""
    i = 0
    n = len(indices)
    while i < n:
        if i == n - 1:
            yield indices[i:i+1]
            break
        chunk_size = random.randint(1, 2)
        yield indices[i:i + chunk_size]
        i += chunk_size

def simulate_random_prediction():
    """Randomly return 0 or 1 for a variant."""
    return random.randint(0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .jsonl")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ---------- load dataset ----------
    df = pd.read_json(args.data, lines=True)

    print("="*80)
    print(f"RANDOM EVAL: {args.data}")
    print("="*80)
    print(f"Total rows (variants): {len(df)}")
    print(f"Unique problems: {df.problem_name.nunique()}")

    # ---------- assign random predictions ----------
    df["prediction"] = [simulate_random_prediction() for _ in range(len(df))]

    # ---------- evaluate per problem, chunked ----------
    accs = []
    for p in df.problem_name.unique():
        df_prob = df[df.problem_name == p]
        idxs = df_prob.index.tolist()

        # chunk variants randomly 1 or 2
        for chunk in chunk_indices(idxs):
            df_chunk = df_prob.loc[chunk]
            accs.append((df_chunk.answer == df_chunk.prediction).all())

    score = round(np.mean(accs) * 100, 4)

    print(f"\nExpected random accuracy: {score}%")
    print(f"Variant-groups solved: {sum(accs)}/{len(accs)}")
    print("="*80)

    # Optional: save predictions
    df.to_json("random_predictions.jsonl", orient="records", lines=True)
    print("Random predictions saved to random_predictions.jsonl")

if __name__ == "__main__":
    main()
