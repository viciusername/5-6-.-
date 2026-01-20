from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import Lab4Params


@dataclass(frozen=True)
class MMCResult:
    lam: float
    mu: float
    c: int
    rho: float
    p0: float
    pw: float
    Lq: float
    Wq: float
    L: float
    W: float


def _erlang_c(c: int, a: float) -> tuple[float, float]:
    """Return (p0, P_wait) for an M/M/c queue.

    a = λ/μ (offered load).
    """
    # p0 denominator
    s = 0.0
    for n in range(c):
        s += (a ** n) / math.factorial(n)
    s += (a ** c) / (math.factorial(c) * (1 - a / c))
    p0 = 1.0 / s

    pw = ((a ** c) / (math.factorial(c) * (1 - a / c))) * p0
    return p0, pw


def mmc_metrics(lam: float, mu: float, c: int) -> MMCResult:
    """Standard steady-state metrics for M/M/c (Erlang C).

    lam: arrival rate (per minute)
    mu: service rate (per minute)
    c: channels
    """
    if mu <= 0 or c <= 0:
        raise ValueError("mu must be >0 and c must be >=1")

    rho = lam / (c * mu)
    if rho >= 1.0:
        # Unstable
        return MMCResult(lam, mu, c, rho, float("nan"), float("nan"), float("inf"), float("inf"), float("inf"), float("inf"))

    a = lam / mu
    p0, pw = _erlang_c(c, a)

    Lq = pw * (rho / (1 - rho))
    Wq = Lq / lam if lam > 0 else 0.0
    L = Lq + a
    W = Wq + 1 / mu

    return MMCResult(lam, mu, c, rho, p0, pw, Lq, Wq, L, W)


def mean_batch_service_time_minutes(p: Lab4Params) -> float:
    """Analytic mean service time for one batch (minutes).

    Uses expectation of each stage. Matches the explanation in the report:
      preprocess: 2*5
      assembly: 4*8
      adjustment: 4*mean(8)
      replacement: 4*(0.02)*3
    """
    n = p.batch_size
    n_pre = int(n * p.preprocess_fraction)

    mean = 0.0
    mean += n_pre * p.preprocess_time_min
    mean += n * p.assembly_time_min
    mean += n * p.adjust_mean_min
    mean += n * p.p_part_replacement * p.replacement_time_min
    return mean


def analytic_table_for_X(p: Lab4Params, X_values: list[float]) -> pd.DataFrame:
    meanS = mean_batch_service_time_minutes(p)
    mu = 1.0 / meanS

    rows = []
    for X in X_values:
        lam = 1.0 / X
        m = mmc_metrics(lam, mu, p.channels)
        rows.append(
            {
                "X_mean_interarrival_min": X,
                "mean_service_time_min": meanS,
                "lambda_per_min": lam,
                "mu_per_min": mu,
                "rho": m.rho,
                "Lq": m.Lq,
                "Wq_min": m.Wq,
                "L": m.L,
                "W_min": m.W,
            }
        )
    return pd.DataFrame(rows)
