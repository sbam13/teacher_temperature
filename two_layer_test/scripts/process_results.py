from __future__ import annotations

import argparse
from pathlib import Path

from two_layer_experiment.results import best_learning_rates, load_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs/two_layer_lr_sweep")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or args.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.runs_dir)
    df.to_csv(out_dir / "tidy_results.csv", index=False)
    best_learning_rates(df).to_csv(out_dir / "best_learning_rates.csv", index=False)


if __name__ == "__main__":
    main()

