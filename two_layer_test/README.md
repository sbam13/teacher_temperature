# Two-Layer Teacher Temperature LR Sweep

This directory contains a self-contained JAX implementation of the finite two-layer teacher/student experiment.

Run locally:

```bash
python3 -m two_layer_experiment.run_lr_sweep --config configs/lr_sweep.yaml
python3 scripts/process_results.py --runs_dir runs/two_layer_lr_sweep
```

Run on slurm:

```bash
sbatch slurm/run_lr_sweep.sbatch
```

The run writes observable CSVs only: per-beta CSVs plus `runs/two_layer_lr_sweep/lr_sweep_results.csv`. Each row contains train/test cross-entropy, train/test logit MSE, train/test teacher and student `h1`/`h2` RMS norms, train/test `student_df_dh1_rms`, and `w1_cosine_fro` at one recorded step. The train and test sets are sampled independently with `p` examples each.

The plotting notebook is `notebooks/pareto_plots.ipynb`. It loads `runs/two_layer_lr_sweep/lr_sweep_results.csv` and writes plot images:

- `runs/two_layer_lr_sweep/pareto_minibatch*_test_xent.png`
- `runs/two_layer_lr_sweep/pareto_population_test_xent.png`

Defaults match the requested experiment: `d=k=100`, `n=300`, `alpha=0.75`, `p=1200`, `steps=1000`, beta log-spaced from `1e-3` to `1e2`, minibatch size derived as `round(p ** delta)` with `delta=0.5`, and population gradient descent. Set `minibatch_size` in the config to override the delta-derived batch size. Ambiguous quantities are configurable: `c` defaults to `0`, `kappa` defaults to `1`, and `g` defaults to `tanh`.

The default grid is intentionally capped at 7 beta values and 11 learning rates to keep the H100 run comfortably under a 1.5 hour upper bound.
