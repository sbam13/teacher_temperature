import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


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


def run_name(condition, lr, seed):
    beta = normalize_beta(condition.get("beta", 1))
    beta_slug = beta.replace(".", "p")
    return (
        f"lr500_{condition['loss_type']}_beta-{beta_slug}_"
        f"{condition['optimizer']}_lr-{slug_float(lr)}_seed{seed}"
    )


def completed(run_dir):
    if (run_dir / "final.pt").exists():
        return True
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return False
    return any('"event": "done"' in line for line in metrics.read_text().splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lr_sweep_500step.yaml")
    ap.add_argument("--skip_completed", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    lrs = cfg.pop("lrs")
    conditions = cfg.pop("conditions")
    out_dir = Path(cfg.get("out_dir", "runs_lr_sweep_500step"))
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_path = out_dir / "failures.jsonl"

    failures = []
    for condition in conditions:
        for lr in lrs:
            name = run_name(condition, lr, cfg.get("seed", 0))
            run_dir = out_dir / name
            if args.skip_completed and completed(run_dir):
                print(f"SKIP completed {name}", flush=True)
                continue

            cmd = [sys.executable, "-m", "tempdistill.train"]
            for key, value in cfg.items():
                if isinstance(value, bool):
                    cmd.append(f"--{key}" if value else f"--no-{key}")
                elif value is not None:
                    cmd.extend([f"--{key}", str(value)])
            cmd.extend(
                [
                    "--loss_type",
                    condition["loss_type"],
                    "--optimizer",
                    condition["optimizer"],
                    "--beta",
                    str(condition.get("beta", 1)),
                    "--lr",
                    str(lr),
                    "--run_name",
                    name,
                ]
            )
            print(shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "run_name": name,
                    "returncode": exc.returncode,
                    "condition": condition,
                    "lr": lr,
                    "cmd": cmd,
                }
                failures.append(row)
                with failure_path.open("a") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"FAILED {name} returncode={exc.returncode}", flush=True)

    if failures:
        print(f"{len(failures)} run(s) failed; see {failure_path}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
