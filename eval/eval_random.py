import argparse
import pandas as pd
import numpy as np
import random

# ---------- helpers ----------
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

    accs = []
    for p in df.problem_name.unique():
        df_prob = df[df.problem_name == p]
        accs.append((df_prob.answer == df_prob.prediction).all())

    score = round(np.mean(accs) * 100, 4)

    print(f"\nExpected random accuracy: {score}%")
    print(f"Problems solved (all variants correct): {sum(accs)}/{len(accs)}")
    print("="*80)

if __name__ == "__main__":
    main()
