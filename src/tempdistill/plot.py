import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_runs(runs_dir):
    rows = []
    for path in Path(runs_dir).glob("*/metrics.jsonl"):
        config_path = path.parent / "config.json"
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["run"] = path.parent.name
            for key in ("loss_type", "optimizer", "beta"):
                row.setdefault(key, config.get(key))
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No metrics.jsonl files found under {runs_dir}")
    df = pd.DataFrame(rows)
    df["beta_label"] = df["beta"].astype(str)
    df["condition"] = df["loss_type"].astype(str) + " beta=" + df["beta_label"] + " " + df["optimizer"].astype(str)
    return df


def lineplot(df, y, out, title=None, event="eval"):
    if y not in df.columns:
        return
    sub = df[(df["event"] == event) & df[y].notna()].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sub, x="step", y=y, hue="condition", units="run", estimator=None, markers=False)
    plt.title(title or y)
    plt.xlabel("Step")
    plt.ylabel(y)
    plt.legend(fontsize="x-small", title=None)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def facet_loss(df, out):
    sub = df[(df["event"] == "eval") & df["distill_eval_loss"].notna()].copy()
    if sub.empty:
        return
    g = sns.relplot(
        data=sub,
        x="step",
        y="distill_eval_loss",
        hue="beta_label",
        col="optimizer",
        row="loss_type",
        kind="line",
        height=3.0,
        aspect=1.25,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Step", "Eval distillation loss")
    g.tight_layout()
    g.savefig(out, dpi=180)
    plt.close("all")


def classification_accuracy_plot(df, out):
    metric_labels = {
        "student_teacher_argmax_acc": "Teacher-argmax accuracy",
        "student_true_acc": "True-token accuracy",
    }
    cols = [c for c in metric_labels if c in df.columns]
    if not cols:
        return
    sub = df[df["event"] == "eval"].copy()
    if sub.empty:
        return
    long = sub.melt(
        id_vars=["step", "condition", "run"],
        value_vars=cols,
        var_name="metric",
        value_name="accuracy",
    )
    long = long[long["accuracy"].notna()]
    if long.empty:
        return
    long["metric"] = long["metric"].map(metric_labels)
    g = sns.relplot(
        data=long,
        x="step",
        y="accuracy",
        hue="condition",
        col="metric",
        kind="line",
        height=3.5,
        aspect=1.35,
        facet_kws={"sharey": True},
        units="run",
        estimator=None,
    )
    g.set_axis_labels("Step", "Eval classification accuracy")
    g.tight_layout()
    g.savefig(out, dpi=180)
    plt.close("all")


def final_scatter(df, out, x="student_fisher_trace_beta", xlabel="Final student Fisher trace proxy at beta"):
    sub = df[df["event"] == "eval"].copy()
    if sub.empty:
        return
    sub = sub.sort_values("step").groupby("run", as_index=False).tail(1)
    if x not in sub.columns:
        return
    sub = sub[sub[x].notna()]
    if sub.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sub,
        x=x,
        y="student_teacher_argmax_acc",
        hue="condition",
        style="optimizer",
        s=90,
    )
    plt.xscale("symlog")
    plt.xlabel(xlabel)
    plt.ylabel("Final teacher-argmax accuracy")
    plt.legend(fontsize="x-small", title=None)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_5000_centered_mse_student_obs")
    ap.add_argument("--out_dir", default="plots_5000_centered_mse_student_obs")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_runs(args.runs_dir)
    df.to_csv(out_dir / "all_metrics.csv", index=False)
    sns.set_theme(style="whitegrid")
    facet_loss(df, out_dir / "distill_loss_grid.png")
    classification_accuracy_plot(df, out_dir / "classification_accuracy.png")
    for y in [
        "hard_xent",
        "hard_teacher_xent",
        "hard_teacher_margin",
        "kl_beta_prime_1",
        "xent_beta_prime_1",
        "logit_mse",
        "centered_logit_mse",
        "log_softmax_mse",
        "student_teacher_argmax_acc",
        "student_true_acc",
        "student_entropy_beta",
        "student_top1_conf_beta",
        "student_effective_support_beta",
        "student_fisher_trace_beta",
        "student_entropy_beta_prime_1",
        "student_fisher_trace_beta_prime_1",
        "teacher_entropy_beta",
        "teacher_fisher_trace_beta",
        "ce_grad_signal_norm_beta",
        "kernel_overlap_initial",
        "kernel_overlap_teacher_target",
        "kernel_effective_rank",
    ]:
        lineplot(df, y, out_dir / f"{y}.png")
    lineplot(df, "grad_norm", out_dir / "grad_norm.png", event="train")
    lineplot(df, "update_norm", out_dir / "update_norm.png", event="train")
    final_scatter(df, out_dir / "final_acc_vs_fisher.png")
    final_scatter(
        df,
        out_dir / "final_acc_vs_teacher_fisher.png",
        x="teacher_fisher_trace_beta",
        xlabel="Teacher target Fisher trace proxy at beta",
    )


if __name__ == "__main__":
    main()
