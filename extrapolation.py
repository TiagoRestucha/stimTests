"""Infinite Distance Extrapolation (IDE) fits — Sec. 3 of Umbrarescu et al.

Ansatz (Eq. 14/16):
    E(d) = A - sum_k B_k exp(-C_k d)
where A = E* is the mitigated (d -> infinity) expectation value.

Richardson (Eq. 17): polynomial in lambda = 1/d, evaluated at lambda = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

A_BOUND = 1.5


def single_exp(d: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a - b * np.exp(-c * d)


def double_exp(d: np.ndarray, a: float, b1: float, c1: float, b2: float, c2: float) -> np.ndarray:
    return a - b1 * np.exp(-c1 * d) - b2 * np.exp(-c2 * d)


@dataclass(frozen=True)
class IdeFit:
    """Result of fitting one ansatz to (d, E_d) data."""

    ansatz: str
    params: tuple[float, ...]
    extrapolated: float  # A = E*
    extrapolated_std: float

    def predict(self, d: np.ndarray) -> np.ndarray:
        fn = single_exp if self.ansatz == "single" else double_exp
        return fn(np.asarray(d, dtype=float), *self.params)


def fit_ide(
    distances: np.ndarray, values: np.ndarray, sigmas: np.ndarray, ansatz: str = "single"
) -> IdeFit:
    """Weighted non-linear least squares with loose bounds A in [-A_BOUND, A_BOUND].

    The paper (Sec. 5.1.1) notes that bounding A to [-1, 1] makes <0|Z|0> = 1
    trivially reachable; we keep A loose so the extrapolation is a fair test.
    """
    d = np.asarray(distances, dtype=float)
    y = np.asarray(values, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if len(d) != len(y) or len(d) != len(s):
        raise ValueError("distances, values and sigmas must have equal length")

    gap = max(float(np.max(y) - np.min(y)), 1e-6)
    if ansatz == "single":
        if len(d) < 3:
            raise ValueError("single-exponential ansatz needs >= 3 points")
        fn, p0 = single_exp, [float(np.max(y)), gap * np.exp(0.3 * d[0]), 0.3]
        lower, upper = [-A_BOUND, 0.0, 0.0], [A_BOUND, np.inf, 5.0]
    elif ansatz == "double":
        if len(d) < 5:
            raise ValueError("double-exponential ansatz needs >= 5 points")
        fn = double_exp
        p0 = [float(np.max(y)), gap * np.exp(0.3 * d[0]), 0.3, gap * np.exp(0.6 * d[0]) * 0.1, 0.8]
        lower, upper = [-A_BOUND, 0.0, 0.0, 0.0, 0.0], [A_BOUND, np.inf, 5.0, np.inf, 5.0]
    else:
        raise ValueError(f"unknown ansatz {ansatz!r}")

    popt, pcov = curve_fit(
        fn, d, y, p0=p0, sigma=s, absolute_sigma=True, bounds=(lower, upper), maxfev=20_000
    )
    a_std = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float("nan")
    return IdeFit(ansatz=ansatz, params=tuple(float(x) for x in popt), extrapolated=float(popt[0]), extrapolated_std=a_std)


def richardson(distances: np.ndarray, values: np.ndarray) -> float:
    """Lagrange extrapolation to lambda = 1/d = 0 (Eq. 17)."""
    lam = 1.0 / np.asarray(distances, dtype=float)
    y = np.asarray(values, dtype=float)
    total = 0.0
    for k, (lk, yk) in enumerate(zip(lam, y)):
        others = np.delete(lam, k)
        total += yk * np.prod(others / (others - lk))
    return float(total)
