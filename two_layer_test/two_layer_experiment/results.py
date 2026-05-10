from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    csv_path = path if path.is_file() else path / "lr_sweep_results.csv"
    df = pd.read_csv(csv_path)
    if "optimizer" not in df.columns:
        df["optimizer"] = "gd"
    return df.sort_values(["optimizer", "mode", "beta", "lr"]).reset_index(drop=True)


def best_learning_rates(df: pd.DataFrame, metric: str = "test_xent") -> pd.DataFrame:
    if "optimizer" not in df.columns:
        df = df.assign(optimizer="gd")
    group_cols = ["optimizer", "mode", "beta", "lr"]
    best_group_cols = ["optimizer", "mode", "beta"]
    final = df[df["step"] == df.groupby(group_cols)["step"].transform("max")]
    idx = final.groupby(best_group_cols)[metric].idxmin()
    cols = ["optimizer", "mode", "beta", "lr", "step", "train_xent", "test_xent", "train_logit_mse", "test_logit_mse"]
    return final.loc[idx, cols].sort_values(best_group_cols).reset_index(drop=True)
