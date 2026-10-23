"""How distinguishable are two devices, as a function of code distance?

Two levels of access are compared:

1. Logical level (only decoded logical outcomes visible — the realistic FTQC
   user view). Each shot is a Bernoulli(p_L) "logical flip".
     * TVD = |p_L,A - p_L,B|
     * shots needed to tell the two devices apart with a two-proportion test
       (significance alpha, power 1 - beta):
         N* = (z_a sqrt(2 pbar(1-pbar)) + z_b sqrt(pA(1-pA) + pB(1-pB)))^2 / (pA - pB)^2
2. Syndrome level (raw detector bits visible — the operator / "forensic" view).
     * per-detector firing rates r_i; shots needed for a combined SNR of
       `snr_target` assuming independent detectors:
         N* = snr_target^2 / sum_i (rA_i - rB_i)^2 / (rA_i(1-rA_i) + rB_i(1-rB_i))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from decoder import LogicalResult


@dataclass(frozen=True)
class LogicalDistinguishability:
    distance: int
    p_a: float
    p_b: float
    tvd: float
    ratio: float
    shots_needed: float


def shots_to_distinguish(p_a: float, p_b: float, alpha: float = 0.01, power: float = 0.9) -> float:
    """Two-proportion z-test sample size per device."""
    if p_a == p_b:
        return math.inf
    z_a = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_b = NormalDist().inv_cdf(power)
    pbar = 0.5 * (p_a + p_b)
    num = z_a * math.sqrt(2.0 * pbar * (1.0 - pbar)) + z_b * math.sqrt(
        p_a * (1.0 - p_a) + p_b * (1.0 - p_b)
    )
    return num**2 / (p_a - p_b) ** 2


def logical_distinguishability(
    results_a: list[LogicalResult], results_b: list[LogicalResult]
) -> list[LogicalDistinguishability]:
    """Pair up results of two devices by distance and compute the metrics."""
    by_d_b = {r.distance: r for r in results_b}
    out = []
    for ra in sorted(results_a, key=lambda r: r.distance):
        rb = by_d_b.get(ra.distance)
        if rb is None:
            continue
        pa, pb = ra.logical_error_rate, rb.logical_error_rate
        out.append(
            LogicalDistinguishability(
                distance=ra.distance,
                p_a=pa,
                p_b=pb,
                tvd=abs(pa - pb),
                ratio=max(pa, pb) / min(pa, pb) if min(pa, pb) > 0 else math.inf,
                shots_needed=shots_to_distinguish(pa, pb),
            )
        )
    return out


def syndrome_shots_needed(rates_a: np.ndarray, rates_b: np.ndarray, snr_target: float = 3.0) -> float:
    """Shots needed to separate two devices from detector firing rates."""
    if rates_a.shape != rates_b.shape:
        raise ValueError("detector rate vectors must have the same shape")
    var = rates_a * (1.0 - rates_a) + rates_b * (1.0 - rates_b)
    mask = var > 0
    information = float(np.sum((rates_a[mask] - rates_b[mask]) ** 2 / var[mask]))
    return math.inf if information == 0.0 else snr_target**2 / information
