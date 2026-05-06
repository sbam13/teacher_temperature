import argparse
import itertools
import math
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def csv_values(text):
    if text is None:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_beta(value):
    if isinstance(value, str) and value.lower() in {"inf", "infty", "infinity"}:
        return "inf"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(parsed):
        return "inf"
    return f"{parsed:g}"


def beta_run_name(value):
    return normalize_beta(value).replace(".", "p")


def completed(run_dir):
    if (run_dir / "final.pt").exists():
        return True
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return False
    return any('"event": "done"' in line for line in metrics.read_text().splitlines())


def planned_jobs(grid, run_order):
    if run_order:
        for item in run_order:
            loss_type = item["loss_type"]
            yield {
                "loss_type": loss_type,
                "optimizer": item["optimizer"],
                "beta": item.get("beta", "1" if loss_type == "mse" else None),
            }
        return

    betas = grid["betas"]
    optimizers = grid["optimizers"]
    loss_types = grid["loss_types"]
    for loss_type, optimizer in itertools.product(loss_types, optimizers):
        beta_values = ["1"] if loss_type == "mse" else betas
        for beta in beta_values:
            yield {"loss_type": loss_type, "optimizer": optimizer, "beta": beta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only_optimizer", default=None)
    ap.add_argument("--loss_type", choices=["xent", "mse"], default=None)
    ap.add_argument("--betas", default=None, help="Comma-separated beta list for xent jobs, e.g. inf,100,10")
    ap.add_argument("--skip_completed", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_order = cfg.pop("run_order", None)
    grid = {
        "betas": cfg.pop("betas"),
        "optimizers": cfg.pop("optimizers"),
        "loss_types": cfg.pop("loss_types", ["xent", "mse"]),
    }
    beta_filter = set(normalize_beta(v) for v in csv_values(args.betas) or [])

    for job in planned_jobs(grid, run_order):
        loss_type = job["loss_type"]
        optimizer = job["optimizer"]
        beta = job["beta"]
        if args.only_optimizer and optimizer != args.only_optimizer:
            continue
        if args.loss_type and loss_type != args.loss_type:
            continue
        if beta_filter and loss_type == "xent" and normalize_beta(beta) not in beta_filter:
            continue

        run_name = f"{loss_type}_beta-{beta_run_name(beta)}_{optimizer}_seed{cfg.get('seed', 0)}"
        run_dir = Path(cfg.get("out_dir", "runs")) / run_name
        if args.skip_completed and completed(run_dir):
            print(f"SKIP completed {run_name}", flush=True)
            continue

        cmd = [sys.executable, "-m", "tempdistill.train"]
        for k, v in cfg.items():
            if isinstance(v, bool):
                cmd.append(f"--{k}" if v else f"--no-{k}")
            elif v is not None:
                cmd.extend([f"--{k}", str(v)])
        cmd.extend(["--loss_type", loss_type, "--optimizer", optimizer, "--beta", str(beta), "--run_name", run_name])
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
