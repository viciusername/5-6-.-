from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lab3Params:
    """Parameters for Lab 3 (single assembly section)."""

    sim_minutes: int = 8 * 60
    mean_interarrival_min: float = 10.0  # exponential mean for batch arrivals
    batch_size: int = 4
    preprocess_fraction: float = 0.5  # half of details need preprocessing

    preprocess_time_min: float = 5.0
    assembly_time_min: float = 8.0

    # Adjustment time: lognormal with mean 8 min
    adjust_mean_min: float = 8.0
    adjust_sigma: float = 0.25  # lognormal sigma (std of ln T)

    p_product_defect: float = 0.01  # defective product found at adjustment
    p_part_replacement: float = 0.02  # replacement needed
    replacement_time_min: float = 3.0

    # Resources (capacities)
    preprocess_servers: int = 1
    assembly_servers: int = 1
    adjust_servers: int = 1


@dataclass(frozen=True)
class Lab4Params:
    """Parameters for Lab 4 (4 parallel sections, one common FIFO queue)."""

    sim_minutes: int = 8 * 60
    batch_size: int = 4
    preprocess_fraction: float = 0.5

    preprocess_time_min: float = 5.0
    assembly_time_min: float = 8.0

    adjust_mean_min: float = 8.0
    adjust_sigma: float = 0.25

    p_product_defect: float = 0.01
    p_part_replacement: float = 0.02
    replacement_time_min: float = 3.0

    channels: int = 4  # four assembly sections

    # Lab4 search settings
    mc_runs: int = 100
    queue_limit: int = 20
    # For the "hard" criterion: at least this fraction of runs must satisfy Qmax <= limit
    pass_fraction: float = 0.95
