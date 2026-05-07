from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Params(NamedTuple):
    w1: jax.Array
    w2: jax.Array


def activation_fn(name: str):
    name = name.lower()
    if name == "tanh":
        return jnp.tanh
    if name == "gelu":
        return jax.nn.gelu
    if name == "erf":
        return jax.scipy.special.erf
    raise ValueError(f"Unknown activation {name!r}; expected tanh, gelu, or erf.")


def activation_prime(name: str, x):
    name = name.lower()
    if name == "tanh":
        y = jnp.tanh(x)
        return 1.0 - y * y
    if name == "gelu":
        sqrt_2 = jnp.sqrt(jnp.asarray(2.0, dtype=x.dtype))
        sqrt_2pi = jnp.sqrt(jnp.asarray(2.0 * jnp.pi, dtype=x.dtype))
        return 0.5 * (1.0 + jax.scipy.special.erf(x / sqrt_2)) + x * jnp.exp(-0.5 * x * x) / sqrt_2pi
    if name == "erf":
        return 2.0 * jnp.exp(-x * x) / jnp.sqrt(jnp.asarray(jnp.pi, dtype=x.dtype))
    raise ValueError(f"Unknown activation {name!r}; expected tanh, gelu, or erf.")


def alpha_s(n: int, alpha: float) -> float:
    total = jnp.sum(jnp.arange(1, n + 1, dtype=jnp.float32) ** (-2.0 * alpha))
    return float(jnp.log(total) / jnp.log(float(n)))


def teacher_exponents(alpha: float, c: float) -> tuple[float, float, float, float]:
    del alpha
    a1 = 0.5
    b1 = 0.0
    a2 = -0.5 * c
    b2 = -c
    return a1, b1, a2, b2


def identity_student_exponents(m: int, c: float) -> tuple[float, float, float, float]:
    del m
    s_identity = 1.0
    a1 = 0.5
    b1 = 0.0
    a2 = 0.25 * (s_identity - 2.0 * c)
    b2 = 0.5 * (s_identity - 2.0 * c)
    return a1, b1, a2, b2


def init_teacher(key, d: int, n: int, k: int, alpha: float, c: float) -> Params:
    _, b1, _, b2 = teacher_exponents(alpha, c)
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (n, d), dtype=jnp.float32) * d ** (-0.5 * b1)
    sigma_sqrt = jnp.arange(1, n + 1, dtype=jnp.float32) ** (-alpha)
    g = jax.random.normal(k2, (k, n), dtype=jnp.float32) * n ** (-0.5 * b2)
    return Params(w1=w1, w2=g * sigma_sqrt[None, :])


def init_student(key, d: int, m: int, k: int, c: float) -> Params:
    _, b1, _, b2 = identity_student_exponents(m, c)
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (m, d), dtype=jnp.float32) * d ** (-0.5 * b1)
    w2 = jax.random.normal(k2, (k, m), dtype=jnp.float32) * m ** (-0.5 * b2)
    return Params(w1=w1, w2=w2)


def forward_teacher(params: Params, x, *, d: int, n: int, c: float, activation: str):
    h1, logits = forward_teacher_parts(params, x, d=d, n=n, c=c, activation=activation)
    return logits


def forward_student(params: Params, x, *, d: int, m: int, c: float, activation: str):
    h1, logits = forward_student_parts(params, x, d=d, m=m, c=c, activation=activation)
    return logits


def forward_teacher_parts(params: Params, x, *, d: int, n: int, c: float, activation: str):
    _, _, a2, _ = teacher_exponents(0.75, c)
    h1 = (x @ params.w1.T) / d**0.5
    logits = (activation_fn(activation)(h1) @ params.w2.T) / (n**c * n**a2)
    return h1, logits


def forward_student_parts(params: Params, x, *, d: int, m: int, c: float, activation: str):
    _, _, a2, _ = identity_student_exponents(m, c)
    h1 = (x @ params.w1.T) / d**0.5
    logits = (activation_fn(activation)(h1) @ params.w2.T) / (m**c * m**a2)
    return h1, logits


def rms(x) -> jax.Array:
    return jnp.sqrt(jnp.mean(jnp.square(x)))
