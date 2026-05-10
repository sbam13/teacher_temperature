from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from two_layer_experiment.config import SweepConfig
from two_layer_experiment.data import sample_population, teacher_diagnostics
from two_layer_experiment.model import init_student, init_teacher
from two_layer_experiment.run_lr_sweep import beta_slug
from two_layer_experiment.train import train_population_lrs


DEFAULT_P_VALUES = (512, 1024, 2048, 4096)
DEFAULT_BETAS = tuple(float(x) for x in range(1, 11))


def transition_lr(
    beta: float,
    *,
    lr_min: float = 0.101,
    lr_max: float = 0.644,
    beta_0: float = 2.27,
    k: float = 6.9,
) -> float:
    exponent = 1.0 / (1.0 + (beta / beta_0) ** (-k / math.log(10.0)))
    return lr_min * (lr_max / lr_min) ** exponent


def load_scaling_config(path: str | Path | None) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) if path is not None else {}
    raw = raw or {}
    sweep_keys = set(SweepConfig.__dataclass_fields__)
    sweep_payload = {key: value for key, value in raw.items() if key in sweep_keys}
    for key in ("betas", "lrs", "modes"):
        if key in sweep_payload and sweep_payload[key] is not None:
            sweep_payload[key] = tuple(sweep_payload[key])
    sweep_config = SweepConfig(**sweep_payload)
    return {
        "sweep_config": sweep_config,
        "p_values": tuple(int(x) for x in raw.get("p_values", DEFAULT_P_VALUES)),
        "betas": tuple(float(x) for x in raw.get("betas", DEFAULT_BETAS)),
        "lr_min": float(raw.get("lr_min", 0.101)),
        "lr_max": float(raw.get("lr_max", 0.644)),
        "beta_0": float(raw.get("beta_0", 2.27)),
        "transition_k": float(raw.get("transition_k", 6.9)),
        "out_dir": str(raw.get("out_dir", sweep_config.out_dir)),
    }


def run_beta_p(config: SweepConfig, beta: float, p: int, lr: float, out_dir: Path) -> pd.DataFrame:
    keys = jax.random.split(jax.random.PRNGKey(config.seed), 4)
    config_for_p = SweepConfig(**{**asdict(config), "p": int(p), "modes": ("population",), "lrs": (float(lr),)})
    m = config_for_p.student_width

    teacher = init_teacher(keys[0], config_for_p.d, config_for_p.n, config_for_p.k, config_for_p.alpha, config_for_p.c)
    init_student_params = init_student(keys[1], config_for_p.d, m, config_for_p.k, config_for_p.c)
    x, y, teacher_logits = sample_population(
        keys[2],
        teacher,
        d=config_for_p.d,
        n=config_for_p.n,
        k=config_for_p.k,
        p=config_for_p.p,
        beta=beta,
        c=config_for_p.c,
        activation=config_for_p.activation,
    )
    test_x, test_y, test_teacher_logits = sample_population(
        keys[3],
        teacher,
        d=config_for_p.d,
        n=config_for_p.n,
        k=config_for_p.k,
        p=config_for_p.p,
        beta=beta,
        c=config_for_p.c,
        activation=config_for_p.activation,
    )

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] beta={beta:g} p={p} lr={lr:.8g}: start",
        flush=True,
    )
    tic = time.time()
    output = train_population_lrs(
        init_student_params,
        jnp.asarray([lr], dtype=jnp.float32),
        x,
        y,
        teacher_logits,
        test_x,
        test_y,
        test_teacher_logits,
        teacher.w1,
        steps=config_for_p.steps,
        observable_every=config_for_p.observable_every,
        d=config_for_p.d,
        m=m,
        c=config_for_p.c,
        activation=config_for_p.activation,
        optimizer="gd",
    )
    seconds = time.time() - tic
    train_diagnostics = teacher_diagnostics(
        teacher,
        x,
        teacher_logits,
        d=config_for_p.d,
        n=config_for_p.n,
        c=config_for_p.c,
    )
    test_diagnostics = teacher_diagnostics(
        teacher,
        test_x,
        test_teacher_logits,
        d=config_for_p.d,
        n=config_for_p.n,
        c=config_for_p.c,
    )

    steps = np.asarray(jax.device_get(output.steps))
    rows = []
    for t_idx, step in enumerate(steps):
        rows.append(
            {
                "task": "FitScalingLaw",
                "beta": float(beta),
                "p": int(p),
                "optimizer": "gd",
                "mode": "population",
                "lr": float(lr),
                "step": int(step),
                "train_xent": float(jax.device_get(output.train_xent[t_idx, 0])),
                "test_xent": float(jax.device_get(output.test_xent[t_idx, 0])),
                "train_logit_mse": float(jax.device_get(output.train_logit_mse[t_idx, 0])),
                "test_logit_mse": float(jax.device_get(output.test_logit_mse[t_idx, 0])),
                "train_teacher_h1_rms": train_diagnostics["teacher_h1_rms"],
                "test_teacher_h1_rms": test_diagnostics["teacher_h1_rms"],
                "train_teacher_h2_rms": train_diagnostics["teacher_h2_rms"],
                "test_teacher_h2_rms": test_diagnostics["teacher_h2_rms"],
                "train_student_h1_rms": float(jax.device_get(output.train_student_h1_rms[t_idx, 0])),
                "test_student_h1_rms": float(jax.device_get(output.test_student_h1_rms[t_idx, 0])),
                "train_student_h2_rms": float(jax.device_get(output.train_student_h2_rms[t_idx, 0])),
                "test_student_h2_rms": float(jax.device_get(output.test_student_h2_rms[t_idx, 0])),
                "train_student_df_dh1_rms": float(jax.device_get(output.train_student_df_dh1_rms[t_idx, 0])),
                "test_student_df_dh1_rms": float(jax.device_get(output.test_student_df_dh1_rms[t_idx, 0])),
                "w1_cosine_fro": float(jax.device_get(output.w1_cosine_fro[t_idx, 0])),
                "seconds": seconds,
            }
        )

    df = pd.DataFrame(rows)
    path = out_dir / f"beta-{beta_slug(beta)}_p-{p}.csv"
    df.to_csv(path, index=False)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] beta={beta:g} p={p}: done in {seconds / 60:.1f} min",
        flush=True,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fit_scaling_law.yaml")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    payload = load_scaling_config(args.config)
    config: SweepConfig = payload["sweep_config"]
    if config.steps % config.observable_every != 0:
        raise ValueError("steps must be divisible by observable_every")
    if config.optimizer != "gd":
        raise ValueError("FitScalingLaw uses full-batch gradient descent; set optimizer: gd.")

    out_dir = Path(args.out_dir or payload["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    configured_p_values = payload["p_values"]
    configured_betas = payload["betas"]
    p_values = (args.p,) if args.p is not None else configured_p_values
    betas = (args.beta,) if args.beta is not None else configured_betas

    metadata = {
        "task": "FitScalingLaw",
        "config": asdict(config),
        "p_values": list(configured_p_values),
        "betas": list(configured_betas),
        "lr_formula": {
            "lr_min": payload["lr_min"],
            "lr_max": payload["lr_max"],
            "beta_0": payload["beta_0"],
            "k": payload["transition_k"],
        },
        "computed_lrs": {
            f"{float(beta):g}": transition_lr(
                float(beta),
                lr_min=payload["lr_min"],
                lr_max=payload["lr_max"],
                beta_0=payload["beta_0"],
                k=payload["transition_k"],
            )
            for beta in configured_betas
        },
    }
    (out_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    if args.beta is not None or args.p is not None:
        selected_metadata = {**metadata, "selected_p_values": list(p_values), "selected_betas": list(betas)}
        selected_name = f"metadata_selected_beta-{beta_slug(float(betas[0]))}.yaml"
        if args.p is not None:
            selected_name = f"metadata_selected_beta-{beta_slug(float(betas[0]))}_p-{int(p_values[0])}.yaml"
        (out_dir / selected_name).write_text(yaml.safe_dump(selected_metadata, sort_keys=False))

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] devices: {jax.devices()}", flush=True)
    print(f"task=FitScalingLaw out_dir={out_dir}", flush=True)
    print(f"p_values={list(p_values)} betas={list(betas)} steps={config.steps}", flush=True)
    if args.dry_run:
        print("dry_run=true; no training launched", flush=True)
        return

    frames = []
    for beta in betas:
        lr = transition_lr(
            float(beta),
            lr_min=payload["lr_min"],
            lr_max=payload["lr_max"],
            beta_0=payload["beta_0"],
            k=payload["transition_k"],
        )
        for p in p_values:
            frames.append(run_beta_p(config, float(beta), int(p), float(lr), out_dir))
    if len(frames) > 1:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "scaling_law_results_partial.csv", index=False)


if __name__ == "__main__":
    main()
