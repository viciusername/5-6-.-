from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RNG:
    """Small wrapper to keep one NumPy Generator per simulation run."""

    seed: int

    def __post_init__(self) -> None:
        self.g = np.random.default_rng(self.seed)

    def exp(self, mean: float) -> float:
        """Exponential with given mean."""
        return float(self.g.exponential(scale=mean))

    def bernoulli(self, p: float) -> bool:
        return bool(self.g.random() < p)

    def lognormal_with_mean(self, mean: float, sigma: float) -> float:
        """Lognormal where E[T] = mean and ln(T) ~ N(mu, sigma^2)."""
        mu = math.log(mean) - 0.5 * sigma * sigma
        return float(self.g.lognormal(mean=mu, sigma=sigma))
