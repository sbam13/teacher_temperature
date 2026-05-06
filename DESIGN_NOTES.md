# Design notes and hypotheses

## Teacher and data

The default teacher is `EleutherAI/pythia-70m-deduped`, which is small enough for
fast H200 inference and was trained on an open corpus, the deduplicated Pile. The
default online data stream is C4 English. This deliberately distills on fresh
streamed minibatches rather than repeatedly cycling a small cached dataset.

## Losses

Finite-temperature xent uses

```text
p_beta(y | x) = softmax(beta * teacher_logits)
q_beta(y | x) = softmax(beta * student_logits)
L = E[-sum_y p_beta(y | x) log q_beta(y | x)]
```

`beta=inf` is implemented as ordinary next-token cross entropy on the observed
token. MSE is untempered logit MSE.

## muP scaling

The student is a local GPT with width-aware initialization and parameter groups.
Its width multiplier is `student_n_embd / student_base_width`. Residual projection
and readout matrix learning rates are divided by this multiplier. This is a
practical muP-style parametrization for a single-file experiment, not a full
coordinate-checking framework.

## Metrics for low-temperature Fisher bottlenecks

The main low-temperature concern is that `softmax(beta * logits)` becomes nearly
one-hot, so only a small part of the vocabulary carries useful probability
signal. The logger includes:

- `teacher_entropy_beta`: low values mean collapsed teacher targets.
- `teacher_effective_support_beta`: `exp(entropy)`, an interpretable support size.
- `teacher_top1_conf_beta`: how hard the softened label has become.
- `teacher_fisher_trace_beta`: `beta^2 * sum_i p_i(1-p_i)`, a categorical Fisher
  trace proxy. This can be nonmonotone in beta because beta increases curvature
  but the distribution also collapses.
- `ce_grad_signal_norm_beta`: `beta * ||q_beta - p_beta||`, a direct proxy for
  softened-label gradient signal.
- `teacher_margin`: large margins predict rapid collapse as beta increases.

Useful post-hoc plots:

- accuracy or logit MSE versus tokens seen, grouped by beta;
- final accuracy versus Fisher trace proxy;
- final logit MSE versus effective support;
- kernel-target overlap versus beta;
- gradient/update norm versus beta for each optimizer.

If the sample complexity worsens only when Fisher trace and effective support
collapse, that supports a target-information bottleneck hypothesis. If MSE avoids
the slowdown at the same logits, that suggests the bottleneck is caused by the
softmax geometry rather than the teacher function itself.

