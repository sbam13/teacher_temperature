from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from two_layer_experiment.config import SweepConfig, load_config
from two_layer_experiment.data import random_batch_indices, sample_population, teacher_diagnostics
from two_layer_experiment.model import init_student, init_teacher
from two_layer_experiment.train import train_minibatch_lrs, train_population_lrs


def beta_slug(beta: float) -> str:
    return f"{beta:g}".replace("-", "m").replace(".", "p")


def run_beta(config: SweepConfig, beta: float, out_dir: Path, progress):
    keys = jax.random.split(jax.random.PRNGKey(config.seed), 5)
    m = config.student_width
    minibatch_size = config.effective_minibatch_size
    teacher = init_teacher(keys[0], config.d, config.n, config.k, config.alpha, config.c)
    init_student_params = init_student(keys[1], config.d, m, config.k, config.c)
    x, y, teacher_logits = sample_population(
        keys[2],
        teacher,
        d=config.d,
        n=config.n,
        k=config.k,
        p=config.p,
        beta=beta,
        c=config.c,
        activation=config.activation,
    )
    test_x, test_y, test_teacher_logits = sample_population(
        keys[3],
        teacher,
        d=config.d,
        n=config.n,
        k=config.k,
        p=config.p,
        beta=beta,
        c=config.c,
        activation=config.activation,
    )
    batch_indices = random_batch_indices(keys[4], steps=config.steps, batch_size=minibatch_size, p=config.p)
    lrs = jnp.asarray(config.lrs, dtype=jnp.float32)

    rows = []
    train_diagnostics = teacher_diagnostics(teacher, x, teacher_logits, d=config.d, n=config.n, c=config.c)
    test_diagnostics = teacher_diagnostics(teacher, test_x, test_teacher_logits, d=config.d, n=config.n, c=config.c)
    for mode in config.modes:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] beta={beta:g} mode={mode}: start", flush=True)
        tic = time.time()
        if mode == "population":
            output = train_population_lrs(
                init_student_params,
                lrs,
                x,
                y,
                teacher_logits,
                test_x,
                test_y,
                test_teacher_logits,
                teacher.w1,
                steps=config.steps,
                observable_every=config.observable_every,
                d=config.d,
                m=m,
                c=config.c,
                activation=config.activation,
                optimizer=config.optimizer,
                adam_beta1=config.adam_beta1,
                adam_beta2=config.adam_beta2,
                adam_eps=config.adam_eps,
                muon_momentum=config.muon_momentum,
                muon_ns_steps=config.muon_ns_steps,
            )
        elif mode.startswith("minibatch"):
            if config.optimizer != "gd":
                raise ValueError("Only full-batch population training supports non-GD optimizers.")
            output = train_minibatch_lrs(
                init_student_params,
                lrs,
                x,
                y,
                teacher_logits,
                test_x,
                test_y,
                test_teacher_logits,
                teacher.w1,
                batch_indices,
                steps=config.steps,
                batch_size=minibatch_size,
                observable_every=config.observable_every,
                d=config.d,
                m=m,
                c=config.c,
                activation=config.activation,
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        steps = np.asarray(jax.device_get(output.steps))
        train_xent = np.asarray(jax.device_get(output.train_xent))
        test_xent = np.asarray(jax.device_get(output.test_xent))
        train_logit_mse = np.asarray(jax.device_get(output.train_logit_mse))
        test_logit_mse = np.asarray(jax.device_get(output.test_logit_mse))
        train_student_h1_rms = np.asarray(jax.device_get(output.train_student_h1_rms))
        test_student_h1_rms = np.asarray(jax.device_get(output.test_student_h1_rms))
        train_student_h2_rms = np.asarray(jax.device_get(output.train_student_h2_rms))
        test_student_h2_rms = np.asarray(jax.device_get(output.test_student_h2_rms))
        train_student_df_dh1_rms = np.asarray(jax.device_get(output.train_student_df_dh1_rms))
        test_student_df_dh1_rms = np.asarray(jax.device_get(output.test_student_df_dh1_rms))
        w1_cosine_fro = np.asarray(jax.device_get(output.w1_cosine_fro))
        seconds = time.time() - tic
        mode_name = f"minibatch{minibatch_size}" if mode.startswith("minibatch") else mode
        for t_idx, step in enumerate(steps):
            for lr_idx, lr in enumerate(config.lrs):
                rows.append(
                    {
                        "beta": float(beta),
                        "optimizer": config.optimizer,
                        "mode": mode_name,
                        "lr": float(lr),
                        "step": int(step),
                        "train_xent": float(train_xent[t_idx, lr_idx]),
                        "test_xent": float(test_xent[t_idx, lr_idx]),
                        "train_logit_mse": float(train_logit_mse[t_idx, lr_idx]),
                        "test_logit_mse": float(test_logit_mse[t_idx, lr_idx]),
                        "train_teacher_h1_rms": train_diagnostics["teacher_h1_rms"],
                        "test_teacher_h1_rms": test_diagnostics["teacher_h1_rms"],
                        "train_teacher_h2_rms": train_diagnostics["teacher_h2_rms"],
                        "test_teacher_h2_rms": test_diagnostics["teacher_h2_rms"],
                        "train_student_h1_rms": float(train_student_h1_rms[t_idx, lr_idx]),
                        "test_student_h1_rms": float(test_student_h1_rms[t_idx, lr_idx]),
                        "train_student_h2_rms": float(train_student_h2_rms[t_idx, lr_idx]),
                        "test_student_h2_rms": float(test_student_h2_rms[t_idx, lr_idx]),
                        "train_student_df_dh1_rms": float(train_student_df_dh1_rms[t_idx, lr_idx]),
                        "test_student_df_dh1_rms": float(test_student_df_dh1_rms[t_idx, lr_idx]),
                        "w1_cosine_fro": float(w1_cosine_fro[t_idx, lr_idx]),
                        "minibatch_size": minibatch_size if mode_name != "population" else config.p,
                        "delta": config.delta,
                        "seconds_for_mode": seconds,
                    }
                )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] beta={beta:g} mode={mode}: "
            f"done in {seconds / 60:.1f} min",
            flush=True,
        )
        progress.update(1)
    df = pd.DataFrame(rows)
    path = out_dir / f"beta-{beta_slug(beta)}.csv"
    df.to_csv(path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--betas", default=None, help="Optional comma-separated beta override.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.betas:
        config = SweepConfig(**{**asdict(config), "betas": tuple(float(x) for x in args.betas.split(","))})
    if config.steps % config.observable_every != 0:
        raise ValueError("steps must be divisible by observable_every")
    out_dir = Path(args.out_dir or config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] devices: {jax.devices()}", flush=True)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting sweep: "
        f"{len(config.betas)} betas x {len(config.lrs)} lrs x {len(config.modes)} modes, "
        f"steps={config.steps}, observable_every={config.observable_every}, "
        f"minibatch_size={config.effective_minibatch_size}, delta={config.delta}, "
        f"optimizer={config.optimizer}",
        flush=True,
    )
    with tqdm(total=len(config.betas) * len(config.modes), desc="beta/mode jobs") as progress:
        for beta in config.betas:
            rows.append(run_beta(config, float(beta), out_dir, progress))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out_dir / "lr_sweep_results.csv", index=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] sweep done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
