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

The run writes observable CSVs only: per-beta CSVs plus `runs/two_layer_lr_sweep/lr_sweep_results.csv`. Each row contains `population_xent`, `population_logit_mse`, teacher/student `h1` and `h2` RMS norms, `student_df_dh1_rms`, and `w1_cosine_fro` at one recorded step.

The plotting notebook is `notebooks/pareto_plots.ipynb`. It loads `runs/two_layer_lr_sweep/lr_sweep_results.csv` and writes plot images:

- `runs/two_layer_lr_sweep/pareto_minibatch32.png`
- `runs/two_layer_lr_sweep/pareto_population.png`

Defaults match the requested experiment: `d=k=100`, `n=300`, `alpha=0.75`, `p=1200`, `steps=100`, beta log-spaced from `1e-3` to `1e2`, minibatch size `32`, and population gradient descent. Ambiguous quantities are configurable: `c` defaults to `0`, `kappa` defaults to `1`, and `g` defaults to `tanh`.

The default grid is intentionally capped at 7 beta values and 11 learning rates to keep the H100 run comfortably under a 1.5 hour upper bound.
