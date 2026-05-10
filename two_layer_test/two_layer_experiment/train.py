from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from two_layer_experiment.model import Params, activation_prime, forward_student, forward_student_parts, identity_student_exponents, rms


class TrainOutput(NamedTuple):
    final_params: Params
    steps: jax.Array
    train_xent: jax.Array
    test_xent: jax.Array
    train_logit_mse: jax.Array
    test_logit_mse: jax.Array
    train_student_h1_rms: jax.Array
    test_student_h1_rms: jax.Array
    train_student_h2_rms: jax.Array
    test_student_h2_rms: jax.Array
    train_student_df_dh1_rms: jax.Array
    test_student_df_dh1_rms: jax.Array
    w1_cosine_fro: jax.Array


def one_hot_xent(logits, labels):
    return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(labels.shape[0]), labels])


def population_loss(params: Params, x, y, *, d: int, m: int, c: float, activation: str):
    logits = forward_student(params, x, d=d, m=m, c=c, activation=activation)
    return one_hot_xent(logits, y)


def population_observables(params: Params, x, y, teacher_logits, teacher_w1, *, d: int, m: int, c: float, activation: str):
    h1, logits = forward_student_parts(params, x, d=d, m=m, c=c, activation=activation)
    _, _, a2, _ = identity_student_exponents(m, c)
    readout_scale = 1.0 / (m**c * m**a2)
    gp2 = jnp.mean(jnp.square(activation_prime(activation, h1)), axis=0)
    w22 = jnp.mean(jnp.square(readout_scale * params.w2), axis=0)
    df_dh1_rms = jnp.sqrt(jnp.mean(gp2 * w22))
    student_norm = params.w1 / jnp.maximum(jnp.linalg.norm(params.w1, axis=1, keepdims=True), 1e-12)
    teacher_norm = teacher_w1 / jnp.maximum(jnp.linalg.norm(teacher_w1, axis=1, keepdims=True), 1e-12)
    cosine = student_norm @ teacher_norm.T
    return (
        one_hot_xent(logits, y),
        jnp.mean(jnp.square(logits - teacher_logits)),
        rms(h1),
        rms((m**c) * logits),
        df_dh1_rms,
        jnp.linalg.norm(cosine, ord="fro"),
    )


def pack_observables(params, x, y, teacher_logits, teacher_w1, *, d: int, m: int, c: float, activation: str):
    return jax.vmap(
        lambda p: population_observables(
            p,
            x,
            y,
            teacher_logits,
            teacher_w1,
            d=d,
            m=m,
            c=c,
            activation=activation,
        )
    )(params)


def pack_train_test_observables(
    params,
    train_x,
    train_y,
    train_teacher_logits,
    test_x,
    test_y,
    test_teacher_logits,
    teacher_w1,
    *,
    d: int,
    m: int,
    c: float,
    activation: str,
):
    train_obs = pack_observables(
        params,
        train_x,
        train_y,
        train_teacher_logits,
        teacher_w1,
        d=d,
        m=m,
        c=c,
        activation=activation,
    )
    test_obs = pack_observables(
        params,
        test_x,
        test_y,
        test_teacher_logits,
        teacher_w1,
        d=d,
        m=m,
        c=c,
        activation=activation,
    )
    return (
        train_obs[0],
        test_obs[0],
        train_obs[1],
        test_obs[1],
        train_obs[2],
        test_obs[2],
        train_obs[3],
        test_obs[3],
        train_obs[4],
        test_obs[4],
        train_obs[5],
    )


def _lr_view(lrs, param):
    return lrs.reshape((-1,) + (1,) * (param.ndim - 1))


def _adam_update(params, grads, state, lrs, step, *, beta1: float, beta2: float, eps: float):
    moments, velocities = state
    moments = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, moments, grads)
    velocities = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g), velocities, grads)
    step_float = step.astype(jnp.float32)
    moments_hat = jax.tree.map(lambda m: m / (1.0 - beta1**step_float), moments)
    velocities_hat = jax.tree.map(lambda v: v / (1.0 - beta2**step_float), velocities)
    params = jax.tree.map(
        lambda p, m, v: p - _lr_view(lrs, p) * m / (jnp.sqrt(v) + eps),
        params,
        moments_hat,
        velocities_hat,
    )
    return params, (moments, velocities)


def _zeropower_via_newton_schulz(update, *, ns_steps: int, eps: float = 1e-7):
    transpose = update.shape[-2] > update.shape[-1]
    if transpose:
        update = jnp.swapaxes(update, -1, -2)
    norm = jnp.linalg.norm(update, axis=(-2, -1), keepdims=True)
    update = update / jnp.maximum(norm, eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        gram = update @ jnp.swapaxes(update, -1, -2)
        update = a * update + (b * gram + c * gram @ gram) @ update
    if transpose:
        update = jnp.swapaxes(update, -1, -2)
    return update


def _muon_update(params, grads, momentum_buffers, lrs, *, momentum: float, ns_steps: int):
    momentum_buffers = jax.tree.map(lambda b, g: momentum * b + g, momentum_buffers, grads)

    def update_param(param, buf):
        direction = _zeropower_via_newton_schulz(buf, ns_steps=ns_steps)
        scale = jnp.sqrt(jnp.maximum(1.0, param.shape[-2] / param.shape[-1]))
        return param - _lr_view(lrs, param) * scale * direction

    params = jax.tree.map(update_param, params, momentum_buffers)
    return params, momentum_buffers


@partial(jax.jit, static_argnames=("steps", "batch_size", "observable_every", "d", "m", "c", "activation"))
def train_minibatch_lrs(
    init_params: Params,
    lrs,
    x,
    y,
    teacher_logits,
    test_x,
    test_y,
    test_teacher_logits,
    teacher_w1,
    batch_indices,
    *,
    steps: int,
    batch_size: int,
    observable_every: int,
    d: int,
    m: int,
    c: float,
    activation: str,
) -> TrainOutput:
    params0 = jax.tree.map(lambda z: jnp.broadcast_to(z, (lrs.shape[0],) + z.shape), init_params)

    def loss_one(params, xb, yb):
        return population_loss(params, xb, yb, d=d, m=m, c=c, activation=activation)

    grad_one = jax.value_and_grad(loss_one)

    def update(params, t):
        idx = batch_indices[t]
        xb = x[idx]
        yb = y[idx]
        _, grads = jax.vmap(grad_one, in_axes=(0, None, None))(params, xb, yb)
        params = jax.tree.map(lambda p, g: p - _lr_view(lrs, p) * g, params, grads)
        return params, None

    def block(params, block_idx):
        offset = block_idx * observable_every
        params, _ = jax.lax.scan(update, params, offset + jnp.arange(observable_every), length=observable_every)
        obs = pack_train_test_observables(
            params,
            x,
            y,
            teacher_logits,
            test_x,
            test_y,
            test_teacher_logits,
            teacher_w1,
            d=d,
            m=m,
            c=c,
            activation=activation,
        )
        return params, obs

    obs0 = pack_train_test_observables(
        params0,
        x,
        y,
        teacher_logits,
        test_x,
        test_y,
        test_teacher_logits,
        teacher_w1,
        d=d,
        m=m,
        c=c,
        activation=activation,
    )
    n_blocks = steps // observable_every
    final_params, obs_scan = jax.lax.scan(block, params0, jnp.arange(n_blocks), length=n_blocks)
    obs = tuple(jnp.concatenate([o0[None, ...], os], axis=0) for o0, os in zip(obs0, obs_scan))
    keep = jnp.arange(0, steps + 1, observable_every)
    return TrainOutput(
        final_params=final_params,
        steps=keep,
        train_xent=obs[0],
        test_xent=obs[1],
        train_logit_mse=obs[2],
        test_logit_mse=obs[3],
        train_student_h1_rms=obs[4],
        test_student_h1_rms=obs[5],
        train_student_h2_rms=obs[6],
        test_student_h2_rms=obs[7],
        train_student_df_dh1_rms=obs[8],
        test_student_df_dh1_rms=obs[9],
        w1_cosine_fro=obs[10],
    )


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "observable_every",
        "d",
        "m",
        "c",
        "activation",
        "optimizer",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "muon_momentum",
        "muon_ns_steps",
    ),
)
def train_population_lrs(
    init_params: Params,
    lrs,
    x,
    y,
    teacher_logits,
    test_x,
    test_y,
    test_teacher_logits,
    teacher_w1,
    *,
    steps: int,
    observable_every: int,
    d: int,
    m: int,
    c: float,
    activation: str,
    optimizer: str = "gd",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
) -> TrainOutput:
    params0 = jax.tree.map(lambda z: jnp.broadcast_to(z, (lrs.shape[0],) + z.shape), init_params)
    optimizer = optimizer.lower()
    if optimizer not in ("gd", "adam", "muon"):
        raise ValueError(f"Unknown optimizer {optimizer!r}; expected gd, adam, or muon.")

    def loss_one(params):
        return population_loss(params, x, y, d=d, m=m, c=c, activation=activation)

    grad_one = jax.value_and_grad(loss_one)

    def update(carry, t):
        params, opt_state = carry
        _, grads = jax.vmap(grad_one)(params)
        if optimizer == "gd":
            params = jax.tree.map(lambda p, g: p - _lr_view(lrs, p) * g, params, grads)
        elif optimizer == "adam":
            params, opt_state = _adam_update(
                params,
                grads,
                opt_state,
                lrs,
                t + 1,
                beta1=adam_beta1,
                beta2=adam_beta2,
                eps=adam_eps,
            )
        elif optimizer == "muon":
            params, opt_state = _muon_update(
                params,
                grads,
                opt_state,
                lrs,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
            )
        return (params, opt_state), None

    def block(carry, block_idx):
        offset = block_idx * observable_every
        carry, _ = jax.lax.scan(update, carry, offset + jnp.arange(observable_every), length=observable_every)
        params, _ = carry
        obs = pack_train_test_observables(
            params,
            x,
            y,
            teacher_logits,
            test_x,
            test_y,
            test_teacher_logits,
            teacher_w1,
            d=d,
            m=m,
            c=c,
            activation=activation,
        )
        return carry, obs

    obs0 = pack_train_test_observables(
        params0,
        x,
        y,
        teacher_logits,
        test_x,
        test_y,
        test_teacher_logits,
        teacher_w1,
        d=d,
        m=m,
        c=c,
        activation=activation,
    )
    n_blocks = steps // observable_every
    if optimizer == "adam":
        state0 = (
            jax.tree.map(jnp.zeros_like, params0),
            jax.tree.map(jnp.zeros_like, params0),
        )
    else:
        state0 = jax.tree.map(jnp.zeros_like, params0)
    (final_params, _), obs_scan = jax.lax.scan(block, (params0, state0), jnp.arange(n_blocks), length=n_blocks)
    obs = tuple(jnp.concatenate([o0[None, ...], os], axis=0) for o0, os in zip(obs0, obs_scan))
    keep = jnp.arange(0, steps + 1, observable_every)
    return TrainOutput(
        final_params=final_params,
        steps=keep,
        train_xent=obs[0],
        test_xent=obs[1],
        train_logit_mse=obs[2],
        test_logit_mse=obs[3],
        train_student_h1_rms=obs[4],
        test_student_h1_rms=obs[5],
        train_student_h2_rms=obs[6],
        test_student_h2_rms=obs[7],
        train_student_df_dh1_rms=obs[8],
        test_student_df_dh1_rms=obs[9],
        w1_cosine_fro=obs[10],
    )
