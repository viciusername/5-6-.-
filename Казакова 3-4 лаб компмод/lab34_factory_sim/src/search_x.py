from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Lab4Params
from .lab4_mc import run_lab4_mc


@dataclass(frozen=True)
class SearchResult:
    X: float
    pass_fraction: float
    qmax_mean: float
    qmax_p95: float


def evaluate_X(p: Lab4Params, X: float, seed0: int = 12345) -> SearchResult:
    df = run_lab4_mc(p, X=X, runs=p.mc_runs, seed0=seed0)
    pass_frac = float((df["max_queue"] <= p.queue_limit).mean())
    qmax_mean = float(df["max_queue"].mean())
    qmax_p95 = float(np.percentile(df["max_queue"], 95))
    return SearchResult(X=X, pass_fraction=pass_frac, qmax_mean=qmax_mean, qmax_p95=qmax_p95)


def find_min_X(p: Lab4Params, X_low: float = 4.0, X_high: float = 40.0, tol: float = 0.5, max_iter: int = 12) -> tuple[SearchResult, pd.DataFrame]:
    """Find minimal X such that pass_fraction >= p.pass_fraction.

    Uses bisection on X with Monte-Carlo evaluation.
    tol is in minutes.
    """

    history = []

    # Ensure upper bound passes
    hi = X_high
    hi_eval = evaluate_X(p, hi)
    history.append(hi_eval.__dict__)
    it_guard = 0
    while hi_eval.pass_fraction < p.pass_fraction and hi < 200:
        hi *= 1.5
        hi_eval = evaluate_X(p, hi)
        history.append(hi_eval.__dict__)
        it_guard += 1
        if it_guard > 6:
            break

    lo = X_low
    lo_eval = evaluate_X(p, lo)
    history.append(lo_eval.__dict__)

    # If even low passes, answer is low
    if lo_eval.pass_fraction >= p.pass_fraction:
        hist_df = pd.DataFrame(history).sort_values("X")
        return lo_eval, hist_df

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_eval = evaluate_X(p, mid)
        history.append(mid_eval.__dict__)

        if mid_eval.pass_fraction >= p.pass_fraction:
            hi = mid
            hi_eval = mid_eval
        else:
            lo = mid
            lo_eval = mid_eval

        if abs(hi - lo) <= tol:
            break

    # Choose best passing (smallest X among pass)
    hist_df = pd.DataFrame(history)
    passing = hist_df[hist_df["pass_fraction"] >= p.pass_fraction].sort_values("X")
    if len(passing) > 0:
        best_row = passing.iloc[0].to_dict()
        best = SearchResult(**{k: best_row[k] for k in SearchResult.__dataclass_fields__.keys()})
        return best, hist_df.sort_values("X")

    # Nothing passed (shouldn't happen if hi ensured)
    worst = SearchResult(X=hi_eval.X, pass_fraction=hi_eval.pass_fraction, qmax_mean=hi_eval.qmax_mean, qmax_p95=hi_eval.qmax_p95)
    return worst, hist_df.sort_values("X")
