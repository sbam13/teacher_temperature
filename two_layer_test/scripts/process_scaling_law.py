from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def load_scaling_results(runs_dir: str | Path) -> pd.DataFrame:
    runs_dir = Path(runs_dir)
    paths = sorted(runs_dir.glob("beta-*_p-*.csv"))
    if not paths:
        aggregate = runs_dir / "scaling_law_results.csv"
        if aggregate.exists():
            return pd.read_csv(aggregate)
        raise FileNotFoundError(f"No beta/p CSV files found in {runs_dir}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def final_losses(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = ["beta", "p"]
    idx = df.groupby(group_cols)["step"].idxmax()
    cols = [
        "beta",
        "p",
        "mode",
        "minibatch_size",
        "target_passes",
        "effective_passes",
        "lr",
        "step",
        "train_xent",
        "test_xent",
        "train_logit_mse",
        "test_logit_mse",
    ]
    keep = [col for col in cols if col in df.columns]
    return df.loc[idx, keep].sort_values(group_cols).reset_index(drop=True)


def fit_scaling_exponents(final: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for beta, group in final.groupby("beta", sort=True):
        group = group.sort_values("p")
        x = np.log(group["p"].to_numpy(dtype=float))
        y = np.log(group[metric].to_numpy(dtype=float))
        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "beta": float(beta),
                "metric": metric,
                "slope": float(slope),
                "scaling_exponent": float(-slope),
                "intercept": float(intercept),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "n_points": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def fit_scaling_law_abc(final: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for beta, group in final.groupby("beta", sort=True):
        group = group.sort_values("p")
        p_vals = group["p"].to_numpy(dtype=float)
        y_vals = group[metric].to_numpy(dtype=float)

        def model(p, a, b, c):
            return np.log(a + b * p**(-c))

        x = np.log(p_vals)
        y_log = np.log(y_vals)
        slope, intercept = np.polyfit(x, y_log, deg=1)
        
        guess_c = max(1e-3, -slope)
        guess_b = np.exp(intercept)
        guess_a = max(1e-8, min(y_vals) * 0.9)

        p0 = [guess_a, guess_b, guess_c]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        try:
            popt, pcov = curve_fit(model, p_vals, y_log, p0=p0, bounds=bounds, maxfev=10000)
            a, b, c = popt
            y_hat = model(p_vals, a, b, c)
            ss_res = float(np.sum((y_log - y_hat) ** 2))
            ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        except Exception:
            a, b, c, r2 = np.nan, np.nan, np.nan, np.nan

        rows.append(
            {
                "beta": float(beta),
                "metric": metric,
                "A": float(a),
                "B": float(b),
                "c": float(c),
                "scaling_exponent": float(c),
                "r2": float(r2),
                "n_points": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs/fit_scaling_law")
    parser.add_argument("--metric", default="test_xent")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    df = load_scaling_results(runs_dir)
    df.to_csv(runs_dir / "scaling_law_results.csv", index=False)
    final = final_losses(df, args.metric)
    final.to_csv(runs_dir / "scaling_law_final_losses.csv", index=False)
    fit_scaling_exponents(final, args.metric).to_csv(runs_dir / "scaling_law_exponents.csv", index=False)
    fit_scaling_law_abc(final, args.metric).to_csv(runs_dir / "scaling_law_exponents_abc.csv", index=False)


if __name__ == "__main__":
    main()
