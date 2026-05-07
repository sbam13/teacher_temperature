from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from two_layer_experiment.model import Params, activation_prime, forward_student, forward_student_parts, identity_student_exponents, rms


class TrainOutput(NamedTuple):
    final_params: Params
    steps: jax.Array
    population_xent: jax.Array
    population_logit_mse: jax.Array
    student_h1_rms: jax.Array
    student_h2_rms: jax.Array
    student_df_dh1_rms: jax.Array
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


@partial(jax.jit, static_argnames=("steps", "batch_size", "observable_every", "d", "m", "c", "activation"))
def train_minibatch_lrs(
    init_params: Params,
    lrs,
    x,
    y,
    teacher_logits,
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
        params = jax.tree.map(lambda p, g: p - lrs.reshape((-1,) + (1,) * (p.ndim - 1)) * g, params, grads)
        return params, None

    def block(params, block_idx):
        offset = block_idx * observable_every
        params, _ = jax.lax.scan(update, params, offset + jnp.arange(observable_every), length=observable_every)
        obs = pack_observables(params, x, y, teacher_logits, teacher_w1, d=d, m=m, c=c, activation=activation)
        return params, obs

    obs0 = pack_observables(params0, x, y, teacher_logits, teacher_w1, d=d, m=m, c=c, activation=activation)
    n_blocks = steps // observable_every
    final_params, obs_scan = jax.lax.scan(block, params0, jnp.arange(n_blocks), length=n_blocks)
    obs = tuple(jnp.concatenate([o0[None, ...], os], axis=0) for o0, os in zip(obs0, obs_scan))
    keep = jnp.arange(0, steps + 1, observable_every)
    return TrainOutput(final_params=final_params, steps=keep, population_xent=obs[0], population_logit_mse=obs[1], student_h1_rms=obs[2], student_h2_rms=obs[3], student_df_dh1_rms=obs[4], w1_cosine_fro=obs[5])


@partial(jax.jit, static_argnames=("steps", "observable_every", "d", "m", "c", "activation"))
def train_population_lrs(
    init_params: Params,
    lrs,
    x,
    y,
    teacher_logits,
    teacher_w1,
    *,
    steps: int,
    observable_every: int,
    d: int,
    m: int,
    c: float,
    activation: str,
) -> TrainOutput:
    params0 = jax.tree.map(lambda z: jnp.broadcast_to(z, (lrs.shape[0],) + z.shape), init_params)

    def loss_one(params):
        return population_loss(params, x, y, d=d, m=m, c=c, activation=activation)

    grad_one = jax.value_and_grad(loss_one)

    def update(params, _):
        _, grads = jax.vmap(grad_one)(params)
        params = jax.tree.map(lambda p, g: p - lrs.reshape((-1,) + (1,) * (p.ndim - 1)) * g, params, grads)
        return params, None

    def block(params, _):
        params, _ = jax.lax.scan(update, params, None, length=observable_every)
        obs = pack_observables(params, x, y, teacher_logits, teacher_w1, d=d, m=m, c=c, activation=activation)
        return params, obs

    obs0 = pack_observables(params0, x, y, teacher_logits, teacher_w1, d=d, m=m, c=c, activation=activation)
    n_blocks = steps // observable_every
    final_params, obs_scan = jax.lax.scan(block, params0, None, length=n_blocks)
    obs = tuple(jnp.concatenate([o0[None, ...], os], axis=0) for o0, os in zip(obs0, obs_scan))
    keep = jnp.arange(0, steps + 1, observable_every)
    return TrainOutput(final_params=final_params, steps=keep, population_xent=obs[0], population_logit_mse=obs[1], student_h1_rms=obs[2], student_h2_rms=obs[3], student_df_dh1_rms=obs[4], w1_cosine_fro=obs[5])
