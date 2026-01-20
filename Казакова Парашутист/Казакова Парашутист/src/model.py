"""Parachutist fall model with quadratic drag (nonlinear) and 2-phase dynamics.

Coordinates:
- Positive direction is DOWN.
- h(t) is altitude (meters) measured UP from ground; therefore dh/dt = -v.

Dynamics in each phase:
    dv/dt = g - k * v^2
    dh/dt = -v

where:
    k = (rho * Cd * A) / (2m)

Phase 1 (freefall): uses (Cd1, A1)
Phase 2 (with parachute): uses (Cd2, A2)

Numerical method: classical Runge–Kutta 4 (RK4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    # Given by task
    m: float  # kg
    H: float  # height, m
    S: float  # half-chest circumference, m
    R: float  # parachute radius, m

    # Physical / model constants
    g: float = 9.81  # m/s^2
    rho: float = 1.225  # kg/m^3
    Cd1: float = 1.0  # body drag coefficient (dimensionless)
    Cd2: float = 1.5  # parachute drag coefficient (dimensionless)
    alpha: float = 0.8  # posture/orientation factor for body area

    # Scenario / simulation settings
    h0: float = 3000.0  # initial altitude, m
    v0: float = 0.0  # initial speed, m/s
    t_open: float = 20.0  # parachute opening time, s
    dt: float = 0.02  # timestep, s
    t_max: float = 600.0  # max simulation time, s

    # "constant speed" criterion
    eps_rel: float = 0.01  # 1% relative band


def body_area(H: float, S: float, alpha: float) -> float:
    """Approximate effective frontal area of the body.

    - S is half-chest circumference (m), so full circumference C = 2S.
    - approximate chest diameter d = C/pi = 2S/pi.
    - approximate frontal area A1 ~ alpha * H * d.

    Units: m^2
    """
    d = (2.0 * S) / np.pi
    return float(alpha * H * d)


def parachute_area(R: float) -> float:
    """Canopy area A2 = pi R^2 (m^2)."""
    return float(np.pi * R * R)


def k_coeff(rho: float, Cd: float, A: float, m: float) -> float:
    """k = rho * Cd * A / (2m), units 1/m."""
    return float(rho * Cd * A / (2.0 * m))


def terminal_velocity(g: float, k: float) -> float:
    """Terminal velocity for dv/dt = g - k v^2 => v_term = sqrt(g/k)."""
    return float(np.sqrt(g / k))


def rk4_step(v: float, h: float, dt: float, g: float, k: float) -> Tuple[float, float]:
    """One RK4 step for the system dv/dt = g - k v^2 ; dh/dt = -v."""

    def f_v(vv: float) -> float:
        return g - k * vv * vv

    def f_h(vv: float) -> float:
        return -vv

    k1_v = f_v(v)
    k1_h = f_h(v)

    v2 = v + 0.5 * dt * k1_v
    k2_v = f_v(v2)
    k2_h = f_h(v2)

    v3 = v + 0.5 * dt * k2_v
    k3_v = f_v(v3)
    k3_h = f_h(v3)

    v4 = v + dt * k3_v
    k4_v = f_v(v4)
    k4_h = f_h(v4)

    v_next = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    h_next = h + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)

    # Guard against tiny negative due to numerics
    return float(max(v_next, 0.0)), float(h_next)


def simulate(p: Params) -> Dict[str, np.ndarray]:
    """Simulate until ground (h<=0) or t_max.

    Returns dict with arrays: t, v, h, phase (1 or 2), v_term (per time).
    """
    A1 = body_area(p.H, p.S, p.alpha)
    A2 = parachute_area(p.R)

    k1 = k_coeff(p.rho, p.Cd1, A1, p.m)
    k2 = k_coeff(p.rho, p.Cd2, A2, p.m)

    vterm1 = terminal_velocity(p.g, k1)
    vterm2 = terminal_velocity(p.g, k2)

    n_steps = int(np.ceil(p.t_max / p.dt)) + 1
    t = np.zeros(n_steps, dtype=float)
    v = np.zeros(n_steps, dtype=float)
    h = np.zeros(n_steps, dtype=float)
    phase = np.zeros(n_steps, dtype=int)
    vterm = np.zeros(n_steps, dtype=float)

    v[0] = p.v0
    h[0] = p.h0
    phase[0] = 1
    vterm[0] = vterm1

    i_last = 0
    for i in range(1, n_steps):
        t[i] = t[i - 1] + p.dt

        is_phase2 = t[i - 1] >= p.t_open
        k = k2 if is_phase2 else k1
        ph = 2 if is_phase2 else 1
        vt = vterm2 if is_phase2 else vterm1

        v[i], h[i] = rk4_step(v[i - 1], h[i - 1], p.dt, p.g, k)
        phase[i] = ph
        vterm[i] = vt

        i_last = i
        if h[i] <= 0.0:
            break

    # trim to actual length
    t = t[: i_last + 1]
    v = v[: i_last + 1]
    h = h[: i_last + 1]
    phase = phase[: i_last + 1]
    vterm = vterm[: i_last + 1]

    return {
        "t": t,
        "v": v,
        "h": h,
        "phase": phase,
        "v_term": vterm,
        "A1": np.array([A1]),
        "A2": np.array([A2]),
        "k1": np.array([k1]),
        "k2": np.array([k2]),
        "vterm1": np.array([vterm1]),
        "vterm2": np.array([vterm2]),
    }


def time_to_constant_speed(t: np.ndarray, v: np.ndarray, v_term: float, eps_rel: float, start_index: int = 0) -> float:
    """Find earliest time index i>=start_index such that for ALL j>=i:
    |v[j]-v_term|/v_term <= eps_rel.

    Returns t[i] if found, else NaN.
    """
    if v_term <= 0:
        return float("nan")

    rel = np.abs(v - v_term) / v_term
    ok = rel <= eps_rel

    # We want earliest i such that ok[i:] all True
    # Compute suffix-AND efficiently
    suffix_all_ok = np.zeros_like(ok, dtype=bool)
    running = True
    for idx in range(len(ok) - 1, -1, -1):
        running = running and bool(ok[idx])
        suffix_all_ok[idx] = running

    candidates = np.where(suffix_all_ok & (np.arange(len(ok)) >= start_index))[0]
    if len(candidates) == 0:
        return float("nan")
    return float(t[int(candidates[0])])
