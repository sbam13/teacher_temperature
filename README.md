# Temperature distillation scaling experiment

This is a small, single-GPU experiment scaffold for online minibatch distillation
from a pretrained LLM teacher into a locally defined muP-scaled GPT student.

Default teacher: `EleutherAI/pythia-70m-deduped`, an Apache-2.0 model trained on
the deduplicated Pile. The default text stream is C4 English. Both can be changed
from the CLI.

## Install

```bash
cd /n/home07/ssainathan/workplace/temperature_experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For an H200, use a recent CUDA PyTorch build. If your environment already has
PyTorch, install the remaining requirements only.

## Quick smoke test

```bash
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
  --weights_out_dir /n/netscratch/pehlevan_lab/Lab/sab
```

## Main sweep

The current resumed H200 config is an expedited AdamW-only pass. Existing
artifacts already cover `beta=0.5` and `beta=1.0`, and the interrupted partial
`beta=4` run should be plotted but not rerun. New runs are capped at 2,000
training steps and still write artifacts to
`runs_5000_centered_mse_student_obs/` so the plotter can combine old and new
metrics. Final student weights are also copied to
`/n/netscratch/pehlevan_lab/Lab/sab`. Teacher logits and token batches are
cached under
`/n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache` and reused across
runs with the same teacher, data, tokenizer, seed, batch size, sequence length,
probe size, precision, and cache dtype.

Run the expedited resumed pass. The config order is MSE AdamW first, then XENT
AdamW `beta=inf`, `beta=100`, and `beta=10`:

```bash
python scripts/run_sweep.py --config configs/h200_sweep.yaml --skip_completed
```

For a dry run of the exact commands:

```bash
python scripts/run_sweep.py --config configs/h200_sweep.yaml --skip_completed --dry_run
```

Outputs are written under `runs_5000_centered_mse_student_obs/<run_name>/`:

- `metrics.jsonl`: scalar training and eval metrics.
- `config.json`: resolved config.
- `final.pt`: final student checkpoint for the run.
- `/n/netscratch/pehlevan_lab/Lab/sab/<run_name>_student_final.pt`: scratch
  copy of the final student weights.
- `/n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache/...`: cached token
  batches and full teacher logits for train and probe/eval batches.

## Plot

```bash
python -m tempdistill.plot \
  --runs_dir runs_5000_centered_mse_student_obs \
  --out_dir plots_5000_centered_mse_student_obs
```

## Instructions for another agent

Goal: execute the temperature distillation experiment on a single H200-class GPU
and return the plots plus a short summary of whether sample complexity changes
with teacher inverse-temperature.

Work from this directory:

```bash
cd /n/home07/ssainathan/workplace/temperature_experiments
```

Set up the environment. Prefer the site CUDA/PyTorch module if one exists; if the
cluster already provides PyTorch, do not reinstall it.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run a minimal smoke test before launching the sweep. This should also verify
that `/n/netscratch/pehlevan_lab/Lab/sab` is writable, because the trainer
copies final student weights there at the end of every run.

```bash
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
```

If the smoke test succeeds, run only the expedited resumed AdamW pass. This
intentionally skips SGD, skips Muon, skips completed `beta=0.5` and `beta=1.0`,
and does not rerun the interrupted `beta=4` run:

```bash
python scripts/run_sweep.py --config configs/h200_sweep.yaml --skip_completed
```

Generate plots:

```bash
python -m tempdistill.plot \
  --runs_dir runs_5000_centered_mse_student_obs \
  --out_dir plots_5000_centered_mse_student_obs
```

Record the run summary in `EXPERIMENT_RESULTS_5000_CENTERED_MSE_STUDENT_OBS.md`.
Before finishing, update this "Instructions for another agent" section with any
major operational issues, workarounds, cluster-specific fixes, cache pitfalls,
or command changes that would save time for the next agent. Also add those notes
to the results markdown so the lesson is preserved with the run artifacts.

Return these artifacts:

- `plots_5000_centered_mse_student_obs/*.png`
- `plots_5000_centered_mse_student_obs/all_metrics.csv`
- `plots_5000_centered_mse_student_obs/classification_accuracy.png`
- each run's `metrics.jsonl`
- each run's `final.pt`
- `/n/netscratch/pehlevan_lab/Lab/sab/*_student_final.pt`
- `/n/netscratch/pehlevan_lab/Lab/sab/teacher_logit_cache/.../manifest.json`
- a concise written summary comparing beta values on loss, logit MSE, accuracy,
  student entropy, student Fisher trace proxy, effective support,
  gradient/update norm, and kernel overlap

Operational notes:

- 2026-05-05 expedited AdamW pass completed on a holygpu H100 node, not an
  H200. No usable PyTorch module was available. `.venv` is a symlink to
  `/tmp/ssainathan/tempdistill-venv`; recreate that target if `/tmp` has been
  cleared.
- On driver `575.57.08` / CUDA `12.9`, the unpinned latest PyTorch wheel
  (`torch 2.11.0+cu130`) failed CUDA initialization. Install
  `torch==2.9.1` after `requirements.txt`, then restore `fsspec==2026.2.0`
  for `datasets`.
- Activate `.venv` before running `scripts/run_sweep.py`; base Python failed on
  missing `yaml` during dry-run.
- Use `HF_HOME=/n/netscratch/pehlevan_lab/Lab/sab/hf_cache` to reuse the
  existing Hugging Face cache. The teacher-logit cache worked in `readwrite`
  mode for the completed pass.
- The expedited sweep artifacts and plots have been generated. The partial
  `xent_beta-4_adamw_seed17` run still has no `final.pt`; leave it partial
  unless a future task explicitly asks to rerun it.
- The first run downloads the teacher and dataset stream from Hugging Face. Set
  `HF_HOME` or `TRANSFORMERS_CACHE` to a scratch/cache filesystem if the home
  directory quota is small.
- If C4 access is unavailable, switch `dataset_name` to `roneneldan/TinyStories`
  and clear `dataset_config`; keep the teacher fixed unless model download fails.
- If the run is too slow, reduce `student_n_layer` to 6 or `student_n_embd` to
  384 in `configs/h200_sweep.yaml`, keeping `student_base_width` fixed at 256.
- Do not compare wall-clock time across beta settings as the main result; compare
  tokens seen or training steps.
- The trainer uses `teacher_cache_mode=readwrite` by default. On a first pass it
  asynchronously prefetches teacher-logit batches in a background thread and
  writes CPU copies of `x`, `y`, and full teacher logits to scratch. Later runs
  load the same cached batches with an order-preserving multi-worker cache
  loader, making training and eval teacher targets reproducible and avoiding
  repeated teacher forward passes. Use
  `teacher_cache_mode=readonly` after the cache is complete to fail fast on
  missing cache entries, or `teacher_cache_mode=off` for debugging.
- The default resumed config uses `teacher_cache_prefetch=8` and
  `teacher_cache_load_workers=4` to overlap cached `.pt` file loads. Reduce
  either value if scratch I/O pressure or host RAM pressure becomes visible.
- Full-vocabulary teacher logits are large. If scratch space or I/O becomes a
  bottleneck, note the issue here before finishing and consider reducing the
  grid, batch size, sequence length, or cache dtype.
- The partial `xent_beta-4_adamw_seed17` run has no `final.pt`; this is expected
  after the interrupted run. Do not rerun it for the expedited pass, but leave
  its `metrics.jsonl` in place so `tempdistill.plot` includes the limited curve.
- The `teacher_entropy_beta` and `teacher_fisher_trace_beta` observables are
  fixed target diagnostics on a fixed probe batch, so they are not expected to
  vary over steps. Use `student_entropy_beta`, `student_fisher_trace_beta`,
  `student_entropy_beta_prime_1`, and `student_fisher_trace_beta_prime_1` for
  student information-geometry curves.

## Metrics included

Primary learning curves:

- finite-temperature distillation cross entropy / KL
- fixed-temperature `beta_prime=1` KL and cross entropy for every run
- next-token hard cross entropy
- hard-teacher cross entropy and hard-teacher margin
- teacher-argmax accuracy
- true-next-token accuracy
- combined eval classification accuracy plot with teacher-argmax and true-token
  accuracy facets
- raw logit MSE, centered logit MSE, and log-softmax MSE

Temperature/Fisher diagnostics:

- student entropy and top-1 confidence at training beta
- student categorical Fisher trace proxy `beta^2 * sum_i q_i(1-q_i)`
- student entropy and Fisher trace at fixed `beta_prime=1`
- teacher entropy and teacher Fisher trace at beta as fixed target diagnostics
- teacher margin and effective support size
- gradient norm and update norm
- CE-gradient signal norm proxy `beta * ||q_student - p_teacher||`

Kernel diagnostics:

- empirical student NTK Gram matrix on a fixed probe batch
- overlap of current kernel with the initial kernel
- overlap of current kernel with a teacher target-similarity kernel

The NTK diagnostic is intentionally small and infrequent. It is meant to show
whether the student is in a lazy-kernel-like regime and whether target geometry
becomes weak or low-rank as beta changes.
