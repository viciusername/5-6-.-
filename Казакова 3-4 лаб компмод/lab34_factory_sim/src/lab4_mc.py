from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import simpy

from .config import Lab4Params
from .random_utils import RNG


@dataclass
class BatchResult:
    batch_id: int
    arrived_at: float
    started_at: float
    finished_at: float
    wait_time: float
    service_time: float
    q_at_arrival: int
    max_queue_seen: int


def _batch_service_time(rng: RNG, p: Lab4Params) -> float:
    """Service time for one batch processed by one section (one server).

    Interprets the section as processing the whole batch sequentially:
    - preprocessing: 2 parts * 5 min
    - assembly: 4 parts * 8 min
    - adjustment: 4 products, lognormal mean 8
      + optional replacement +3 min with p=0.02 per product (only if product not defective)

    Defective product (p=0.01) does not reduce time because adjustment already happened.
    """
    n = p.batch_size
    n_pre = int(n * p.preprocess_fraction)

    total = 0.0
    total += n_pre * p.preprocess_time_min
    total += n * p.assembly_time_min

    for _ in range(n):
        total += rng.lognormal_with_mean(p.adjust_mean_min, p.adjust_sigma)
        is_defect = rng.bernoulli(p.p_product_defect)
        if not is_defect and rng.bernoulli(p.p_part_replacement):
            total += p.replacement_time_min

    return total


class Lab4MC:
    def __init__(self, env: simpy.Environment, p: Lab4Params, rng: RNG, mean_interarrival_min: float):
        self.env = env
        self.p = p
        self.rng = rng
        self.mean_interarrival = mean_interarrival_min

        self.sections = simpy.Resource(env, capacity=p.channels)
        self.results: List[BatchResult] = []

        self._batch_counter = 0
        self.max_queue = 0

    def run(self) -> None:
        self.env.process(self._arrival_process())

    def _update_max_queue(self) -> None:
        q = len(self.sections.queue)
        if q > self.max_queue:
            self.max_queue = q

    def _arrival_process(self):
        while self.env.now < self.p.sim_minutes:
            inter = self.rng.exp(self.mean_interarrival)
            yield self.env.timeout(inter)
            if self.env.now >= self.p.sim_minutes:
                break

            batch_id = self._batch_counter
            self._batch_counter += 1

            q_at_arrival = len(self.sections.queue)
            self._update_max_queue()

            self.env.process(self._one_batch(batch_id, q_at_arrival))

    def _one_batch(self, batch_id: int, q_at_arrival: int):
        arrived = self.env.now

        with self.sections.request() as req:
            yield req
            started = self.env.now
            wait = started - arrived
            self._update_max_queue()

            service = _batch_service_time(self.rng, self.p)
            yield self.env.timeout(service)

            finished = self.env.now
            self._update_max_queue()

            self.results.append(
                BatchResult(
                    batch_id=batch_id,
                    arrived_at=arrived,
                    started_at=started,
                    finished_at=finished,
                    wait_time=wait,
                    service_time=service,
                    q_at_arrival=q_at_arrival,
                    max_queue_seen=self.max_queue,
                )
            )


def run_lab4_once(seed: int, p: Lab4Params, mean_interarrival_min: float) -> Tuple[pd.DataFrame, dict]:
    env = simpy.Environment()
    rng = RNG(seed)
    model = Lab4MC(env, p, rng, mean_interarrival_min)
    model.run()
    env.run(until=p.sim_minutes)

    df = pd.DataFrame([r.__dict__ for r in model.results])

    summary = {
        "X_mean_interarrival_min": mean_interarrival_min,
        "sim_minutes": float(p.sim_minutes),
        "batches_arrived": model._batch_counter,
        "batches_finished": len(model.results),
        "mean_wait_min": float(df["wait_time"].mean()) if not df.empty else 0.0,
        "mean_service_min": float(df["service_time"].mean()) if not df.empty else 0.0,
        "mean_system_time_min": float((df["wait_time"] + df["service_time"]).mean()) if not df.empty else 0.0,
        "max_queue": int(model.max_queue),
    }
    return df, summary


def run_lab4_mc(p: Lab4Params, X: float, runs: int, seed0: int = 12345) -> pd.DataFrame:
    """Run many Monte-Carlo replications for Lab4 and return per-run summary table."""
    rows = []
    for i in range(runs):
        _, summ = run_lab4_once(seed0 + i, p, X)
        summ["run"] = i + 1
        rows.append(summ)
    return pd.DataFrame(rows)
