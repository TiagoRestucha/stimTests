"""Monte-Carlo sampling + MWPM decoding (Stim, PyMatching, Sinter)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sinter

from surface_memory import DeviceNoise, noisy_memory_circuit


@dataclass(frozen=True)
class LogicalResult:
    """Decoded statistics of one (device, distance) memory experiment."""

    device: str
    distance: int
    shots: int
    errors: int

    @property
    def logical_error_rate(self) -> float:
        return self.errors / self.shots if self.shots else math.nan

    @property
    def expectation(self) -> float:
        """<Z_L> = 1 - 2 p_L (Eq. for trivial EVs, Sec. 2.2.4)."""
        return 1.0 - 2.0 * self.logical_error_rate

    @property
    def expectation_std(self) -> float:
        """Binomial standard error of <Z_L>: 2 sqrt(p(1-p)/N)."""
        p = self.logical_error_rate
        return 2.0 * math.sqrt(max(p * (1.0 - p), 1.0 / self.shots) / self.shots)


def collect_logical(
    devices: list[DeviceNoise],
    distances: list[int],
    max_shots: int = 1_000_000,
    max_errors: int = 5_000,
    num_workers: int = 4,
) -> list[LogicalResult]:
    """Sample + decode every (device, distance) pair in parallel with Sinter."""
    tasks = [
        sinter.Task(
            circuit=noisy_memory_circuit(d, dev),
            json_metadata={"device": dev.name, "d": d},
        )
        for dev in devices
        for d in distances
    ]
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=max_shots,
        max_errors=max_errors,
    )
    results = [
        LogicalResult(
            device=s.json_metadata["device"],
            distance=int(s.json_metadata["d"]),
            shots=s.shots,
            errors=s.errors,
        )
        for s in stats
    ]
    return sorted(results, key=lambda r: (r.device, r.distance))


def detector_firing_rates(device: DeviceNoise, distance: int, shots: int, seed: int = 0) -> np.ndarray:
    """Mean firing probability of every detector (the *physical-level* signal)."""
    circuit = noisy_memory_circuit(distance, device)
    dets = circuit.compile_detector_sampler(seed=seed).sample(shots)
    return dets.mean(axis=0)
