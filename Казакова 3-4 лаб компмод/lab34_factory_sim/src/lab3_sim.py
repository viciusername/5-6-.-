from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import simpy

from .config import Lab3Params
from .random_utils import RNG


@dataclass
class PartResult:
    batch_id: int
    part_id: int
    arrived_at: float
    finished_at: float
    lead_time: float
    is_good: bool
    had_replacement: bool


@dataclass
class ResourceStats:
    name: str
    busy_time: float = 0.0

    def utilization(self, sim_time: float, capacity: int) -> float:
        if sim_time <= 0 or capacity <= 0:
            return 0.0
        return self.busy_time / (sim_time * capacity)


class Lab3Model:
    def __init__(self, env: simpy.Environment, params: Lab3Params, rng: RNG) -> None:
        self.env = env
        self.p = params
        self.rng = rng

        self.preprocess = simpy.Resource(env, capacity=params.preprocess_servers)
        self.assembly = simpy.Resource(env, capacity=params.assembly_servers)
        self.adjust = simpy.Resource(env, capacity=params.adjust_servers)

        self.stats_pre = ResourceStats("preprocess")
        self.stats_asm = ResourceStats("assembly")
        self.stats_adj = ResourceStats("adjust")

        self.results: List[PartResult] = []
        self._batch_counter = 0

    def run(self) -> None:
        self.env.process(self._arrival_process())

    def _arrival_process(self):
        while self.env.now < self.p.sim_minutes:
            inter = self.rng.exp(self.p.mean_interarrival_min)
            yield self.env.timeout(inter)
            if self.env.now >= self.p.sim_minutes:
                break
            batch_id = self._batch_counter
            self._batch_counter += 1
            # create 4 parts; first half need preprocessing
            need_pre = [True] * int(self.p.batch_size * self.p.preprocess_fraction) + [False] * (
                self.p.batch_size - int(self.p.batch_size * self.p.preprocess_fraction)
            )
            for part_id in range(self.p.batch_size):
                self.env.process(self._part_flow(batch_id, part_id, need_pre[part_id]))

    def _service(self, res: simpy.Resource, duration: float, stat: ResourceStats):
        with res.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(duration)
            stat.busy_time += self.env.now - start

    def _part_flow(self, batch_id: int, part_id: int, needs_preprocess: bool):
        arrived = self.env.now

        if needs_preprocess:
            yield from self._service(self.preprocess, self.p.preprocess_time_min, self.stats_pre)

        yield from self._service(self.assembly, self.p.assembly_time_min, self.stats_asm)

        # Adjustment is lognormal with mean 8
        adj_time = self.rng.lognormal_with_mean(self.p.adjust_mean_min, self.p.adjust_sigma)
        yield from self._service(self.adjust, adj_time, self.stats_adj)

        # Quality outcomes
        is_defect = self.rng.bernoulli(self.p.p_product_defect)
        had_repl = False
        if not is_defect:
            if self.rng.bernoulli(self.p.p_part_replacement):
                had_repl = True
                # replacement takes additional time (assume done at adjustment post)
                yield from self._service(self.adjust, self.p.replacement_time_min, self.stats_adj)

        finished = self.env.now
        self.results.append(
            PartResult(
                batch_id=batch_id,
                part_id=part_id,
                arrived_at=arrived,
                finished_at=finished,
                lead_time=finished - arrived,
                is_good=not is_defect,
                had_replacement=had_repl,
            )
        )


def run_lab3(seed: int, params: Lab3Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run one Lab3 simulation and return (parts_df, summary_df)."""

    env = simpy.Environment()
    rng = RNG(seed)
    model = Lab3Model(env, params, rng)
    model.run()
    env.run(until=params.sim_minutes)

    parts_df = pd.DataFrame([r.__dict__ for r in model.results])

    sim_t = float(params.sim_minutes)
    summary = {
        "sim_minutes": sim_t,
        "batches_arrived": model._batch_counter,
        "parts_finished": len(model.results),
        "good_parts": int(parts_df["is_good"].sum()) if not parts_df.empty else 0,
        "defective_parts": int((~parts_df["is_good"]).sum()) if not parts_df.empty else 0,
        "mean_lead_time_min": float(parts_df["lead_time"].mean()) if not parts_df.empty else 0.0,
        "util_preprocess": model.stats_pre.utilization(sim_t, params.preprocess_servers),
        "util_assembly": model.stats_asm.utilization(sim_t, params.assembly_servers),
        "util_adjust": model.stats_adj.utilization(sim_t, params.adjust_servers),
        "replacement_count": int(parts_df["had_replacement"].sum()) if not parts_df.empty else 0,
    }
    summary_df = pd.DataFrame([summary])
    return parts_df, summary_df
