from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    csv_path = path if path.is_file() else path / "lr_sweep_results.csv"
    df = pd.read_csv(csv_path)
    return df.sort_values(["mode", "beta", "lr"]).reset_index(drop=True)


def best_learning_rates(df: pd.DataFrame) -> pd.DataFrame:
    final = df[df["step"] == df.groupby(["mode", "beta", "lr"])["step"].transform("max")]
    idx = final.groupby(["mode", "beta"])["population_xent"].idxmin()
    cols = ["mode", "beta", "lr", "step", "population_xent", "population_logit_mse"]
    return final.loc[idx, cols].sort_values(["mode", "beta"]).reset_index(drop=True)
