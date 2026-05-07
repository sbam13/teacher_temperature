from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

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
    steps: int = 100
    observable_every: int = 5
    minibatch_size: int = 32
    seed: int = 0
    activation: str = "tanh"
    betas: tuple[float, ...] = tuple(float(x) for x in __import__("numpy").logspace(-3, 2, 7))
    lrs: tuple[float, ...] = tuple(float(x) for x in __import__("numpy").logspace(-4, 0, 11))
    modes: tuple[str, ...] = ("minibatch32", "population")
    out_dir: str = "runs/two_layer_lr_sweep"

    @property
    def student_width(self) -> int:
        return int(round(self.kappa * self.n))


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
