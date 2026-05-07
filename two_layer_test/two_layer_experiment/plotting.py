from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def pareto_plot(df: pd.DataFrame, mode: str, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    subset = df[df["mode"] == mode].copy()
    subset = subset[subset["step"] == subset.groupby(["beta", "lr"])["step"].transform("max")]
    for beta, group in subset.groupby("beta", sort=True):
        group = group.sort_values("lr")
        ax.plot(group["lr"], group["population_xent"], marker="o", linewidth=1.6, label=f"beta={beta:g}")
        best = group.loc[group["population_xent"].idxmin()]
        ax.scatter(best["lr"], best["population_xent"], marker="*", s=140, color="black", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("SGD learning rate")
    ax.set_ylabel("Population cross-entropy after 100 steps")
    ax.set_title("Minibatch SGD, batch size 32" if mode == "minibatch32" else "Population gradient descent")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title=None, ncols=2, fontsize="small")
    return ax


def time_plot(df: pd.DataFrame, mode: str, beta: float, lr: float, metrics: list[str], ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    subset = df[(df["mode"] == mode) & (df["beta"] == beta) & (df["lr"] == lr)].sort_values("step")
    for metric in metrics:
        ax.plot(subset["step"], subset[metric], marker="o", linewidth=1.5, label=metric)
    ax.set_xlabel("SGD step")
    ax.grid(True, alpha=0.25)
    ax.legend(title=None, fontsize="small")
    return ax
