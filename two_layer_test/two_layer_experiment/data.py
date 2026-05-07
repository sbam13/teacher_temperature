from __future__ import annotations

import jax
import jax.numpy as jnp

from two_layer_experiment.model import Params, forward_teacher, rms


def sample_population(key, teacher: Params, *, d: int, n: int, k: int, p: int, beta: float, c: float, activation: str):
    kx, kg = jax.random.split(key)
    x = jax.random.normal(kx, (p, d), dtype=jnp.float32)
    logits = forward_teacher(teacher, x, d=d, n=n, c=c, activation=activation)
    gumbel = jax.random.gumbel(kg, (p, k), dtype=jnp.float32)
    y = jnp.argmax(beta * logits + gumbel, axis=-1).astype(jnp.int32)
    return x, y, logits


def random_batch_indices(key, *, steps: int, batch_size: int, p: int):
    return jax.random.randint(key, (steps, batch_size), minval=0, maxval=p)


def teacher_diagnostics(teacher: Params, x, logits, *, d: int, n: int, c: float):
    h1 = (x @ teacher.w1.T) / d**0.5
    return {
        "teacher_h1_rms": float(rms(h1)),
        "teacher_h2_rms": float(rms((n**c) * logits)),
    }
