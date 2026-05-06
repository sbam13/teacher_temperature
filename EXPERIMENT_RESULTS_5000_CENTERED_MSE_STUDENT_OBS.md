# Resumed Centered-MSE Student-Observable Experiment Results

Created: 2026-05-05

This file tracks the resumed AdamW pass using the objective and observable fixes
implemented after `EXPERIMENT_RESULTS.md`. The output directory name still says
`5000` because it already contains the completed `beta=0.5` and `beta=1.0`
AdamW runs plus the interrupted partial `beta=4` run.

## Experiment Definition

- Config: `configs/h200_sweep.yaml`
- Steps per new resumed run: 2,000
- Already completed: `xent beta=0.5 adamw` and `xent beta=1.0 adamw` at 5,000
  steps
- Include but do not rerun: interrupted `xent beta=4 adamw`, currently through
  train step 1,450 with last eval at step 1,400 and no `final.pt`
- Run directory: `runs_5000_centered_mse_student_obs/`
- Plot directory: `plots_5000_centered_mse_student_obs/`
- Scratch student weights: `/n/netscratch/pehlevan_lab/Lab/sab/*_student_final.pt`
- Teacher-logit cache: `/n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache/`
- Teacher: `EleutherAI/pythia-70m-deduped`
- Dataset: C4 English train stream
- Student: local muP GPT, 8 layers, width 512, 8 heads

## Implemented Fixes

- Finite-beta XENT now logs and optimizes the normalized objective
  `-(p_teacher_beta * log q_student_beta).sum() / beta`.
- `beta=inf` now uses the hard-teacher limit of the same normalized objective:
  `max_j student_logit_j - student_logit_teacher_argmax`.
- MSE distillation now optimizes centered logit MSE, removing the per-token
  additive logit gauge.
- Evaluation logs raw logit MSE, centered logit MSE, and log-softmax MSE.
- Evaluation logs fixed-temperature `beta_prime=1` KL and XENT for every run.
- Student observables now include entropy, top-1 confidence, effective support,
  and Fisher trace at the training beta and at fixed `beta_prime=1`.
- Teacher entropy and teacher Fisher trace remain fixed target diagnostics on
  the fixed probe batch and are not expected to vary over steps.
- Training batches use an asynchronous teacher-logit prefetch/cache path by
  default. The cache stores CPU copies of `x`, `y`, and full teacher logits for
  reproducibility and reuse across runs with the same cache key. Cached runs now
  use an order-preserving multi-worker loader so several cached `.pt` batches
  can be loaded ahead while training consumes the previous batches.
- Eval/probe teacher logits are cached once in the same cache key directory.
- Plotting now emits `classification_accuracy.png`, a faceted eval plot for
  teacher-argmax accuracy and true-token accuracy.

## Execution Commands

```bash
cd /n/home07/ssainathan/workplace/temperature_experiments

python -m tempdistill.train \
  --max_steps 5 \
  --eval_every 5 \
  --batch_size 2 \
  --seq_len 64 \
  --student_n_layer 2 \
  --student_n_embd 128 \
  --student_n_head 4 \
  --loss_type xent \
  --beta 1 \
  --optimizer adamw \
  --run_name smoke_5000_script_check \
  --out_dir runs_5000_centered_mse_student_obs \
  --weights_out_dir /n/netscratch/pehlevan_lab/Lab/sab \
  --teacher_cache_dir /n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache \
  --teacher_cache_mode readwrite \
  --teacher_cache_prefetch 2

python scripts/run_sweep.py --config configs/h200_sweep.yaml --skip_completed

python -m tempdistill.plot \
  --runs_dir runs_5000_centered_mse_student_obs \
  --out_dir plots_5000_centered_mse_student_obs
```

## Results Summary

Quick check completed for existing `xent beta=0.5 adamw` and
`xent beta=1.0 adamw` curves:

- Plots:
  `quick_check_beta_0p5_1_plots/train_curves_beta_0p5_1.png`

  ![Quick check train curves](quick_check_beta_0p5_1_plots/train_curves_beta_0p5_1.png)

  `quick_check_beta_0p5_1_plots/eval_observables_beta_0p5_1.png`

  ![Quick check eval observables](quick_check_beta_0p5_1_plots/eval_observables_beta_0p5_1.png)
- Summary CSV: `quick_check_beta_0p5_1_plots/summary_beta_0p5_1.csv`
- The two completed curves are intelligible: distillation loss, KL, hard XENT,
  and accuracies improve; gradient and update norms remain finite; teacher
  observables stay constant on the fixed probe batch; student entropy/effective
  support decrease as expected.
- `centered_logit_mse` is large but finite and mostly flat, from about
  `10770` at initialization to `10524` for `beta=0.5` and `10681` for `beta=1.0`
  at step 4,999. The cached probe teacher logits have centered RMS about
  `103.78`, so the teacher's own centered second moment is about `10770`; this
  metric is therefore a blunt full-vocabulary logit-shape diagnostic, not a
  primary learning signal.

The resumed 2,000-step MSE/`beta=inf`/`beta=100`/`beta=10` jobs completed on
2026-05-05 with `--skip_completed`. The run used a holygpu H100 node, not an
H200. Plots and the consolidated CSV were regenerated in
`plots_5000_centered_mse_student_obs/`.

Generated artifacts:

- Per-run logs and final weights under `runs_5000_centered_mse_student_obs/`
  for MSE, `beta=inf`, `beta=100`, and `beta=10`, plus prior completed
  `beta=0.5` and `beta=1.0` runs.
- Scratch final weights under
  `/n/netscratch/pehlevan_lab/Lab/sab/*_student_final.pt` for every completed
  run. The interrupted `xent_beta-4_adamw_seed17` run still has no `final.pt`.
- Teacher cache manifests under
  `/n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache/.../manifest.json`.
- `plots_5000_centered_mse_student_obs/all_metrics.csv` and the PNG plot suite,
  including `classification_accuracy.png`.

Beta comparison:

- Loss: the normalized training-beta distillation loss decreases as beta gets
  sharper, but that value is not directly comparable across temperatures. The
  fixed `beta_prime=1` KL is more comparable: final KL is best for `beta=1`
  (`1.9213`), then `beta=0.5` (`2.6921`), then the partial `beta=4` curve
  (`4.7192`), MSE (`4.6107`), `beta=10` (`5.7554`), `beta=inf` (`5.8467`), and
  `beta=100` (`6.6015`).
- Logit MSE: centered/log-softmax MSE stay near the teacher-centered second
  moment for XENT runs (`~10524` to `10770` centered MSE), so they are blunt
  diagnostics for XENT. MSE directly optimizes the centered gauge-fixed logits
  and reaches much lower centered MSE (`211.3418`) and log-softmax MSE
  (`243.5977`).
- Accuracy: final true-token accuracy is strongest for `beta=1` over 5,000
  steps (`0.1924`) and `beta=10` over 2,000 steps (`0.1904`). The partial
  `beta=4` run reaches `0.1670` true accuracy by eval step 1,400. `beta=100`
  and `beta=inf` underperform, and MSE has high teacher-argmax agreement
  (`0.3330`) but poor true-token accuracy (`0.0371`).
- Entropy, Fisher, and support: at the training beta, `beta=0.5` remains very
  diffuse (entropy `9.6894`, support `16646.6`, Fisher `0.2499`), `beta=1` to
  `beta=10` become progressively sharper, `beta=100` has a very large Fisher
  proxy (`9328.9971`), and `beta=inf` is the hard one-hot limit
  (entropy/Fisher `0`, support `1`).
- Gradient/update norms: latest train grad norms are modest for finite XENT
  (`0.1610` to `0.4979`) except hard `beta=inf` (`5.5856`). MSE has much larger
  raw gradients (`46.2855`). Update norms are around `0.07` to `0.18` for
  completed jobs, while the interrupted `beta=4` row shows a larger latest
  update norm (`0.9526`) from its earlier schedule state.
- Kernel overlap: kernel overlap with initialization remains high
  (`0.9379` to `0.99099`), suggesting the runs stay close to their initial NTK.
  Teacher-target overlap is highest for `beta=0.5` among finite soft-XENT runs
  (`0.5334`) and decreases with sharper beta; `beta=inf` has near-perfect
  target overlap by construction of the hard target kernel diagnostic.

## Operational Notes for Future Agents

- Do not run the SGD sweep.
- Do not run Muon in this expedited resumed pass unless explicitly requested.
- The expedited resumed command has already completed:
  `python scripts/run_sweep.py --config configs/h200_sweep.yaml --skip_completed`.
  Rerun only to regenerate intentionally or after changing the config.
- The plot command has already completed:
  `python -m tempdistill.plot --runs_dir runs_5000_centered_mse_student_obs --out_dir plots_5000_centered_mse_student_obs`.
- Keep the partial `xent_beta-4_adamw_seed17/metrics.jsonl` in place so plots
  include that limited curve; do not expect a `final.pt` for it.
- This host had no usable PyTorch module. `.venv` points at
  `/tmp/ssainathan/tempdistill-venv`; rebuild that target if `/tmp` was cleared.
- Do not let `pip` install the latest PyTorch wheel on driver `575.57.08` /
  CUDA `12.9`; `torch 2.11.0+cu130` failed CUDA initialization. `torch==2.9.1`
  (`+cu128`) worked, and `fsspec==2026.2.0` was restored afterward for
  `datasets`.
- Activate `.venv` before invoking `scripts/run_sweep.py`; base Python failed
  on missing `yaml` during dry-run.
- Use `HF_HOME=/n/netscratch/pehlevan_lab/Lab/sab/hf_cache` to avoid home-quota
  pressure and reuse the existing Hugging Face model/dataset cache.
- If cache loading remains slow, tune `teacher_cache_prefetch` and
  `teacher_cache_load_workers`; current defaults are `8` and `4`.

### Final Metrics

From the latest eval row of each `metrics.jsonl`; `beta=4` is intentionally
partial.

| loss | optimizer | beta | eval step | distill loss | KL beta prime 1 | teacher-argmax acc | true acc | centered logit MSE | log-softmax MSE | student entropy beta | student Fisher beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xent | adamw | 0.5 | 4999 | 19.3952 | 2.6921 | 0.1914 | 0.1299 | 10523.7549 | 10653.8750 | 9.6894 | 0.2499 |
| xent | adamw | 1 | 4999 | 6.0345 | 1.9213 | 0.2266 | 0.1924 | 10681.3359 | 10845.2090 | 6.2844 | 0.9550 |
| xent | adamw | 4 partial | 1400 | 1.4850 | 4.7192 | 0.2129 | 0.1670 | 10742.0879 | 11002.6055 | 6.0066 | 15.2052 |
| xent | adamw | 10 | 1999 | 0.5622 | 5.7554 | 0.1982 | 0.1904 | 10758.3750 | 11026.0840 | 5.7635 | 94.8501 |
| xent | adamw | 100 | 1999 | 0.0649 | 6.6015 | 0.1504 | 0.1328 | 10759.4883 | 11028.5391 | 6.1247 | 9328.9971 |
| xent | adamw | inf | 1999 | 0.0336 | 5.8467 | 0.0723 | 0.0186 | 10769.7480 | 11036.0713 | 0.0000 | 0.0000 |
| mse | adamw | 1 | 1999 | 8.7238 | 4.6107 | 0.3330 | 0.0371 | 211.3418 | 243.5977 | 7.3498 | 0.9956 |

Additional diagnostics from the latest eval row plus latest train row:

| loss | beta | effective support beta | CE-grad signal beta | train grad norm | train update norm | kernel overlap init | kernel overlap target | kernel effective rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xent | 0.5 | 16646.5996 | 0.0176 | 0.1610 | 0.1209 | 0.9784 | 0.5334 | 5.4597 |
| xent | 1 | 1007.9031 | 0.2635 | 0.4979 | 0.1144 | 0.9379 | 0.3916 | 6.6497 |
| xent | 4 partial | 722.8672 | 1.3508 | 0.2838 | 0.9526 | 0.9910 | 0.3609 | 5.0983 |
| xent | 10 | 522.0306 | 3.2698 | 0.1729 | 0.1112 | 0.9883 | 0.3124 | 4.8099 |
| xent | 100 | 861.9523 | 36.3769 | 0.3535 | 0.0790 | 0.9662 | 0.2859 | 3.2000 |
| xent | inf | 1.0000 | 1.3120 | 5.5856 | 0.0693 | 0.9661 | 0.9999 | 3.1997 |
| mse | 1 | 1555.9554 | 0.2981 | 46.2855 | 0.1802 | 0.9627 | 0.2796 | 3.1755 |

## Plot Index

Generated plot outputs are embedded below.

- `plots_5000_centered_mse_student_obs/distill_loss_grid.png`

  ![Distill loss grid](plots_5000_centered_mse_student_obs/distill_loss_grid.png)

- `plots_5000_centered_mse_student_obs/classification_accuracy.png`

  ![Classification accuracy](plots_5000_centered_mse_student_obs/classification_accuracy.png)

- `plots_5000_centered_mse_student_obs/kl_beta_prime_1.png`

  ![KL beta prime 1](plots_5000_centered_mse_student_obs/kl_beta_prime_1.png)

- `plots_5000_centered_mse_student_obs/xent_beta_prime_1.png`

  ![XENT beta prime 1](plots_5000_centered_mse_student_obs/xent_beta_prime_1.png)

- `plots_5000_centered_mse_student_obs/hard_xent.png`

  ![Hard XENT](plots_5000_centered_mse_student_obs/hard_xent.png)

- `plots_5000_centered_mse_student_obs/hard_teacher_xent.png`

  ![Hard teacher XENT](plots_5000_centered_mse_student_obs/hard_teacher_xent.png)

- `plots_5000_centered_mse_student_obs/hard_teacher_margin.png`

  ![Hard teacher margin](plots_5000_centered_mse_student_obs/hard_teacher_margin.png)

- `plots_5000_centered_mse_student_obs/logit_mse.png`

  ![Logit MSE](plots_5000_centered_mse_student_obs/logit_mse.png)

- `plots_5000_centered_mse_student_obs/centered_logit_mse.png`

  ![Centered logit MSE](plots_5000_centered_mse_student_obs/centered_logit_mse.png)

- `plots_5000_centered_mse_student_obs/log_softmax_mse.png`

  ![Log-softmax MSE](plots_5000_centered_mse_student_obs/log_softmax_mse.png)

- `plots_5000_centered_mse_student_obs/student_teacher_argmax_acc.png`

  ![Student teacher-argmax accuracy](plots_5000_centered_mse_student_obs/student_teacher_argmax_acc.png)

- `plots_5000_centered_mse_student_obs/student_true_acc.png`

  ![Student true accuracy](plots_5000_centered_mse_student_obs/student_true_acc.png)

- `plots_5000_centered_mse_student_obs/student_entropy_beta.png`

  ![Student entropy beta](plots_5000_centered_mse_student_obs/student_entropy_beta.png)

- `plots_5000_centered_mse_student_obs/student_fisher_trace_beta.png`

  ![Student Fisher trace beta](plots_5000_centered_mse_student_obs/student_fisher_trace_beta.png)

- `plots_5000_centered_mse_student_obs/student_entropy_beta_prime_1.png`

  ![Student entropy beta prime 1](plots_5000_centered_mse_student_obs/student_entropy_beta_prime_1.png)

- `plots_5000_centered_mse_student_obs/student_fisher_trace_beta_prime_1.png`

  ![Student Fisher trace beta prime 1](plots_5000_centered_mse_student_obs/student_fisher_trace_beta_prime_1.png)

- `plots_5000_centered_mse_student_obs/teacher_entropy_beta.png`

  ![Teacher entropy beta](plots_5000_centered_mse_student_obs/teacher_entropy_beta.png)

- `plots_5000_centered_mse_student_obs/teacher_fisher_trace_beta.png`

  ![Teacher Fisher trace beta](plots_5000_centered_mse_student_obs/teacher_fisher_trace_beta.png)

- `plots_5000_centered_mse_student_obs/final_acc_vs_fisher.png`

  ![Final accuracy vs Fisher](plots_5000_centered_mse_student_obs/final_acc_vs_fisher.png)

- `plots_5000_centered_mse_student_obs/final_acc_vs_teacher_fisher.png`

  ![Final accuracy vs teacher Fisher](plots_5000_centered_mse_student_obs/final_acc_vs_teacher_fisher.png)
