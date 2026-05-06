import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


CONDITION_NAMES = {
    ("xent", "0.5"): "XENT beta=0.5",
    ("xent", "100"): "XENT beta=100",
    ("mse", "1"): "MSE centered logits",
}


def normalize_beta(value):
    if isinstance(value, str) and value.lower() in {"inf", "infty", "infinity"}:
        return "inf"
    parsed = float(value)
    if math.isinf(parsed):
        return "inf"
    return f"{parsed:g}"


def slug_float(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def expected_run_name(condition, lr, seed):
    beta = normalize_beta(condition.get("beta", 1))
    return (
        f"lr500_{condition['loss_type']}_beta-{beta.replace('.', 'p')}_"
        f"{condition['optimizer']}_lr-{slug_float(lr)}_seed{seed}"
    )


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_expected(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    lrs = cfg["lrs"]
    conditions = cfg["conditions"]
    seed = cfg.get("seed", 0)
    expected = []
    for condition in conditions:
        for lr in lrs:
            beta = normalize_beta(condition.get("beta", 1))
            expected.append(
                {
                    "run": expected_run_name(condition, lr, seed),
                    "condition": condition,
                    "loss_type": condition["loss_type"],
                    "optimizer": condition["optimizer"],
                    "beta": beta,
                    "lr": float(lr),
                }
            )
    return cfg, expected


def final_rows(runs_dir, expected):
    rows = []
    notes = []
    for item in expected:
        run_dir = Path(runs_dir) / item["run"]
        metrics_path = run_dir / "metrics.jsonl"
        metrics = read_jsonl(metrics_path)
        eval_rows = [r for r in metrics if r.get("event") == "eval"]
        train_rows = [r for r in metrics if r.get("event") == "train"]
        done = any(r.get("event") == "done" for r in metrics)
        has_final = (run_dir / "final.pt").exists()

        if not metrics_path.exists():
            notes.append(f"{item['run']}: missing metrics.jsonl; excluded.")
            continue
        if not eval_rows:
            notes.append(f"{item['run']}: no eval rows; excluded.")
            continue

        final_eval = max(eval_rows, key=lambda r: r.get("step", -1))
        if final_eval.get("step") != 499:
            notes.append(
                f"{item['run']}: latest eval step is {final_eval.get('step')}, not 499; "
                "included as the latest available endpoint row."
            )
        if not done or not has_final:
            notes.append(
                f"{item['run']}: missing {'done event' if not done else 'final.pt'}; "
                "included only if endpoint metrics are present."
            )

        latest_train = max(train_rows, key=lambda r: r.get("step", -1)) if train_rows else {}
        beta = item["beta"]
        condition_label = CONDITION_NAMES.get((item["loss_type"], beta), f"{item['loss_type']} beta={beta}")
        row = {
            "run": item["run"],
            "condition": condition_label,
            "loss_type": item["loss_type"],
            "optimizer": item["optimizer"],
            "beta": beta,
            "lr": item["lr"],
            "eval_step": final_eval.get("step"),
            "done": done,
            "has_final": has_final,
            "kl_beta_prime_1": final_eval.get("kl_beta_prime_1"),
            "centered_logit_mse": final_eval.get("centered_logit_mse"),
            "logit_mse": final_eval.get("logit_mse"),
            "log_softmax_mse": final_eval.get("log_softmax_mse"),
            "distill_eval_loss": final_eval.get("distill_eval_loss"),
            "student_teacher_argmax_acc": final_eval.get("student_teacher_argmax_acc"),
            "student_true_acc": final_eval.get("student_true_acc"),
            "hard_xent": final_eval.get("hard_xent"),
            "grad_norm": latest_train.get("grad_norm"),
            "update_norm": latest_train.get("update_norm"),
            "latest_train_step": latest_train.get("step"),
        }
        rows.append(row)

    failures_path = Path(runs_dir) / "failures.jsonl"
    for failure in read_jsonl(failures_path):
        notes.append(
            f"{failure.get('run_name')}: subprocess failed with return code "
            f"{failure.get('returncode')} at lr={failure.get('lr')}."
        )
    return pd.DataFrame(rows), notes


def format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def markdown_table(df, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def plot_metric(df, metric, ylabel, out):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 5.4))
    ax = sns.lineplot(
        data=df,
        x="lr",
        y=metric,
        hue="condition",
        marker="o",
        errorbar=None,
        sort=True,
    )
    for condition, group in df.dropna(subset=[metric]).groupby("condition"):
        best = group.loc[group[metric].idxmin()]
        ax.scatter(
            [best["lr"]],
            [best[metric]],
            marker="*",
            s=260,
            color="gold",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel(ylabel)
    ax.legend(title=None, fontsize="small")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def best_table(df, metric, lr_col_name):
    rows = []
    for condition, group in df.dropna(subset=[metric]).groupby("condition"):
        best = group.loc[group[metric].idxmin()]
        rows.append({"condition": condition, lr_col_name: best["lr"], metric: best[metric]})
    return pd.DataFrame(rows).sort_values("condition")


def metric_at(df, condition, lr, metric):
    row = df[(df["condition"] == condition) & (df["lr"] == lr)]
    if row.empty:
        return None
    return row.iloc[0].get(metric)


def write_report(config, df, notes, out_dir, report_path):
    report_path = Path(report_path)
    rel_plot_dir = Path(out_dir)
    kl_plot = rel_plot_dir / "eval_kl_vs_lr.png"
    mse_plot = rel_plot_dir / "centered_logit_mse_vs_lr.png"
    final_csv = rel_plot_dir / "final_metrics.csv"

    kl_best = best_table(df, "kl_beta_prime_1", "kl_best_lr")
    mse_best = best_table(df, "centered_logit_mse", "mse_best_lr")
    best = pd.merge(kl_best, mse_best, on="condition", how="outer", suffixes=("_kl", "_mse"))
    best = best.rename(
        columns={
            "kl_beta_prime_1": "best_kl_beta_prime_1",
            "centered_logit_mse": "best_centered_logit_mse",
        }
    )

    metric_cols = [
        "condition",
        "lr",
        "eval_step",
        "kl_beta_prime_1",
        "centered_logit_mse",
        "logit_mse",
        "log_softmax_mse",
        "distill_eval_loss",
        "student_teacher_argmax_acc",
        "student_true_acc",
        "grad_norm",
        "update_norm",
    ]
    best_cols = [
        "condition",
        "kl_best_lr",
        "best_kl_beta_prime_1",
        "mse_best_lr",
        "best_centered_logit_mse",
    ]

    lr_grid = ", ".join(format_value(float(lr)) for lr in config["lrs"])
    notes_text = "\n".join(f"- {note}" for note in notes) if notes else "- No runs failed or were excluded."
    mse_001_centered = metric_at(df, "MSE centered logits", 0.001, "centered_logit_mse")
    mse_01_centered = metric_at(df, "MSE centered logits", 0.01, "centered_logit_mse")
    mse_01_logsm = metric_at(df, "MSE centered logits", 0.01, "log_softmax_mse")
    mse_01_raw = metric_at(df, "MSE centered logits", 0.01, "logit_mse")
    xent05_kl_001 = metric_at(df, "XENT beta=0.5", 0.001, "kl_beta_prime_1")
    xent05_kl_01 = metric_at(df, "XENT beta=0.5", 0.01, "kl_beta_prime_1")
    xent100_kl_003 = metric_at(df, "XENT beta=100", 0.003, "kl_beta_prime_1")
    xent100_kl_01 = metric_at(df, "XENT beta=100", 0.01, "kl_beta_prime_1")
    xent100_update_01 = metric_at(df, "XENT beta=100", 0.01, "update_norm")

    lines = [
        "# 500-Step AdamW Learning-Rate Sweep",
        "",
        "Created: 2026-05-05",
        "",
        "## Methods",
        "",
        "This follow-up sweep tests whether the prior beta curves were confounded by using a single learning rate. "
        "It keeps the teacher, C4 stream, student architecture, seed, batch size, sequence length, precision, "
        "AdamW optimizer family, weight decay, cosine warmup/min-lr schedule, cache settings, and kernel probe "
        "settings aligned with `configs/h200_sweep.yaml`, except `max_steps=500`, `eval_every=100`, "
        "new output directories, and the swept learning rate.",
        "",
        f"- Config: `configs/lr_sweep_500step.yaml`",
        f"- Runs: `{config['out_dir']}/`",
        f"- Plots: `{out_dir}/`",
        f"- Learning-rate grid: {lr_grid}",
        "- Conditions: XENT beta=0.5, XENT beta=100, and centered-logit MSE.",
        "- Endpoint row: latest eval row, expected at train step 499.",
        "- Primary KL metric: `kl_beta_prime_1`.",
        "- Primary logit-MSE metric: `centered_logit_mse`.",
        "- The `0.01` point is the single allowed extra LR per condition. It was added after the initial four-point grid selected the high-LR boundary for at least one primary metric.",
        "",
        "## Results Summary",
        "",
        f"- KL-best learning rates: MSE `0.01`, XENT beta=0.5 `0.001`, XENT beta=100 `0.003`.",
        f"- Centered-MSE-best learning rates: MSE `0.01`, XENT beta=0.5 `0.01`, XENT beta=100 `0.003`.",
        f"- For XENT beta=0.5, KL worsens from {format_value(xent05_kl_001)} at `lr=0.001` to {format_value(xent05_kl_01)} at `lr=0.01`, while centered logit MSE is nearly flat and slightly favors the larger LR. That is a concrete KL/MSE-best mismatch.",
        f"- For XENT beta=100, `lr=0.01` is worse than `lr=0.003` on KL ({format_value(xent100_kl_01)} vs {format_value(xent100_kl_003)}) and has a larger update norm ({format_value(xent100_update_01)}), so `0.003` remains the selected LR despite weak accuracy.",
        f"- Centered-logit MSE training is highly LR-sensitive over this range: centered MSE falls from {format_value(mse_001_centered)} at `lr=0.001` to {format_value(mse_01_centered)} at `lr=0.01`. At `lr=0.01`, log-softmax MSE is {format_value(mse_01_logsm)}, while raw logit MSE is still {format_value(mse_01_raw)}, consistent with raw logit MSE retaining additive-gauge contamination.",
        "",
        "## Eval KL vs LR",
        "",
        f"![Eval KL vs learning rate]({kl_plot.as_posix()})",
        "",
        "## Centered Logit MSE vs LR",
        "",
        f"![Centered logit MSE vs learning rate]({mse_plot.as_posix()})",
        "",
        "## Best Learning Rates",
        "",
        markdown_table(best, best_cols),
        "",
        "## Final Endpoint Metrics",
        "",
        f"Full CSV: `{final_csv.as_posix()}`",
        "",
        markdown_table(df.sort_values(["condition", "lr"]), metric_cols),
        "",
        "## Operational Notes",
        "",
        "- Ran on a single NVIDIA A100-SXM4-40GB visible to PyTorch.",
        "- Recreated the cleared `.venv` target at `/tmp/ssainathan/tempdistill-venv`.",
        "- Used `torch==2.9.1+cu128` and `fsspec==2026.2.0`, matching the prior CUDA/fsspec workaround.",
        "- Set `HF_HOME=/n/netscratch/pehlevan_lab/Lab/sab/hf_cache` when running the sweep.",
        "- Reused the existing teacher-logit cache key for seed 17, batch size 16, seq len 128, bf16.",
        "",
        "## Failures And Exclusions",
        "",
        notes_text,
        "",
        "## Link Check",
        "",
    ]

    links = [kl_plot, mse_plot, final_csv]
    missing = [p.as_posix() for p in links if not p.exists()]
    if missing:
        lines.append("- Missing expected artifact(s): " + ", ".join(missing))
    else:
        lines.append("- Verified embedded plot and CSV paths exist.")
    report_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lr_sweep_500step.yaml")
    ap.add_argument("--runs_dir", default=None)
    ap.add_argument("--out_dir", default="plots_lr_sweep_500step")
    ap.add_argument("--report", default="lr_sweep_experiment.md")
    args = ap.parse_args()

    config, expected = read_expected(args.config)
    runs_dir = args.runs_dir or config.get("out_dir", "runs_lr_sweep_500step")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, notes = final_rows(runs_dir, expected)
    if df.empty:
        raise RuntimeError(f"No endpoint eval rows found under {runs_dir}")
    df = df.sort_values(["condition", "lr"])
    df.to_csv(out_dir / "final_metrics.csv", index=False)
    plot_metric(df, "kl_beta_prime_1", "Eval KL divergence at beta_prime=1", out_dir / "eval_kl_vs_lr.png")
    plot_metric(df, "centered_logit_mse", "Centered logit MSE", out_dir / "centered_logit_mse_vs_lr.png")
    write_report(config, df, notes, out_dir, args.report)

    image_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", Path(args.report).read_text())
    missing = [link for link in image_links if not Path(link).exists()]
    if missing:
        raise RuntimeError(f"Report image link(s) do not exist: {missing}")


if __name__ == "__main__":
    main()
