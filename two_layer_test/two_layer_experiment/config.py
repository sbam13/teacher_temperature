from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class SweepConfig:
    d: int = 100
    k: int = 100
    n: int = 300
    kappa: float = 1.0
    alpha: float = 0.75
    c: float = 0.0
    p: int = 1200
    delta: Optional[float] = 0.5
    steps: int = 100
    observable_every: int = 5
    minibatch_size: Optional[int] = None
    seed: int = 0
    activation: str = "tanh"
    optimizer: str = "gd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    betas: tuple[float, ...] = tuple(float(x) for x in __import__("numpy").logspace(-3, 2, 7))
    lrs: tuple[float, ...] = tuple(float(x) for x in __import__("numpy").logspace(-4, 0, 11))
    modes: tuple[str, ...] = ("minibatch", "population")
    out_dir: str = "runs/two_layer_lr_sweep"

    @property
    def student_width(self) -> int:
        return int(round(self.kappa * self.n))

    @property
    def effective_minibatch_size(self) -> int:
        if self.minibatch_size is not None:
            return int(self.minibatch_size)
        if self.delta is None:
            raise ValueError("Either minibatch_size or delta must be set for minibatch SGD.")
        return max(1, int(round(self.p ** self.delta)))


def load_config(path: str | Path | None) -> SweepConfig:
    if path is None:
        return SweepConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("betas", "lrs", "modes"):
        if key in raw:
            raw[key] = tuple(raw[key])
    return SweepConfig(**raw)


def save_config(config: SweepConfig, path: str | Path) -> None:
    payload = asdict(config)
    payload["betas"] = list(config.betas)
    payload["lrs"] = list(config.lrs)
    payload["modes"] = list(config.modes)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
