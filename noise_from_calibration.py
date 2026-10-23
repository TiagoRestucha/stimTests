"""Calibration-driven noise model for Stim circuits (CSV loader + noise insertion).

Reads the per-qubit and per-pair calibration CSVs exported from the IBM fake
backends and turns an ideal Stim circuit into a noisy one.

    cal = CalibratedNoise.from_csv("fake_kingston_qubits.csv",
                                   "fake_kingston_two_qubit_gates.csv", seed=0)
    noisy = cal.apply(ideal_circuit)

Modelling choices (all deliberate, change them via the fields of CalibratedNoise):
  * Topology is ignored (all-to-all). Each code qubit / each CX-CZ pair draws one
    calibration row from the device's pooled data with a FIXED seed, so a device
    is one reproducible, spatially inhomogeneous realisation at every distance.
  * Non-operational entries are dropped (2q error >= 0.5, readout >= 0.5,
    missing or non-positive T1/T2/1q error).
  * Gate infidelity r (average gate error) -> depolarizing probability
    p = r (d+1)/d, i.e. 1.5 r for 1 qubit and 1.25 r for 2 qubits.
  * CZ data is used as the CX error ("close enough").
  * Readout: symmetric `readout_error` flip BEFORE every measurement.
  * Reset / preparation error = 0 by default (`p_reset`), rz is virtual (no noise),
    Z/S/S_DAG are virtual too. frequency/anharmonicity are not used.
  * Idle noise: Pauli-twirled T1/T2 decay, applied at the end of every layer
    (every TICK). A layer lasts as long as its slowest operation, and a qubit is
    charged only for the time it was NOT busy in that layer. Circuits without
    TICKs therefore get no idle noise.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import stim

CALIBRATION_DIR = Path("/home/tiagorestucha/Workspaces/qiskit")
MAX_USABLE_ERROR = 0.5  # entries at or above this are treated as non-operational

# --- instruction families ---------------------------------------------------
ANNOTATIONS = {
    "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "MPAD",
}
VIRTUAL_1Q = {"I", "Z", "S", "S_DAG"}  # frame changes, no noise, no time
PHYSICAL_1Q = {
    "H", "X", "Y", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "H_XY", "H_YZ",
}
TWO_Q = {"CX", "CZ", "CY", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SWAP", "ISWAP"}
RESETS = {"R": "X_ERROR", "RX": "Z_ERROR"}
MEASURES = {  # name -> (flip error placed before the measurement, resets too?)
    "M": ("X_ERROR", False),
    "MR": ("X_ERROR", True),
    "MX": ("Z_ERROR", False),
    "MRX": ("Z_ERROR", True),
}


# --- data ---------------------------------------------------------------------
def _num(row: dict, key: str) -> float:
    """Float from a CSV cell, NaN if missing/empty."""
    try:
        v = row.get(key)
        return float("nan") if v in (None, "") else float(v)
    except ValueError:
        return float("nan")


def _check_seconds(name: str, value: float, lo: float, hi: float) -> None:
    """Fail loudly if a time looks like us/ns instead of seconds."""
    if not (lo <= value <= hi):
        raise ValueError(
            f"{name}={value!r} is outside the plausible range [{lo}, {hi}] seconds; "
            "is it in microseconds or nanoseconds? Convert to seconds first."
        )


@dataclass(frozen=True)
class QubitCal:
    t1: float  # s
    t2: float  # s (clipped to <= 2 T1)
    readout_error: float
    gate1_error: float  # sx_error, fallback x_error, then id_error
    gate1_duration: float  # s
    measure_duration: float  # s


@dataclass(frozen=True)
class PairCal:
    error: float
    duration: float  # s


@dataclass
class DeviceCalibration:
    """Pooled, filtered calibration data of one device."""

    name: str
    qubits: list[QubitCal]
    pairs: list[PairCal]
    dropped_qubits: int = 0
    dropped_pairs: int = 0

    @classmethod
    def from_rows(
        cls, qubit_rows: list[dict], pair_rows: list[dict], name: str = "device"
    ) -> "DeviceCalibration":
        qubits, dropped_q = [], 0
        for r in qubit_rows:
            t1, t2 = _num(r, "T1"), _num(r, "T2")
            ro = _num(r, "readout_error")
            if math.isnan(ro):  # fall back to the mean of the two assignment errors
                ro = np.nanmean([_num(r, "prob_meas0_prep1"), _num(r, "prob_meas1_prep0")])
            g1 = next(
                (v for v in (_num(r, k) for k in ("sx_error", "x_error", "id_error"))
                 if not math.isnan(v)),
                float("nan"),
            )
            g1_dur = next(
                (v for v in (_num(r, k) for k in ("sx_duration", "x_duration"))
                 if not math.isnan(v)),
                float("nan"),
            )
            m_dur = next(
                (v for v in (_num(r, k) for k in ("measure_duration", "readout_length"))
                 if not math.isnan(v)),
                float("nan"),
            )
            needed = (t1, t2, ro, g1, g1_dur, m_dur)
            if (any(math.isnan(v) for v in needed) or t1 <= 0 or t2 <= 0
                    or ro >= MAX_USABLE_ERROR or g1 >= MAX_USABLE_ERROR or g1 < 0 or ro < 0):
                dropped_q += 1
                continue
            _check_seconds("T1", t1, 1e-7, 0.1)
            _check_seconds("T2", t2, 1e-7, 0.1)
            _check_seconds("1q gate duration", g1_dur, 1e-10, 1e-5)
            _check_seconds("measure duration", m_dur, 1e-9, 1e-3)
            qubits.append(QubitCal(t1, min(t2, 2 * t1), ro, g1, g1_dur, m_dur))

        pairs, dropped_p, seen = [], 0, set()
        for r in pair_rows:
            a, b = int(r["qubit_a"]), int(r["qubit_b"])
            key = (min(a, b), max(a, b))
            if key in seen:  # CSVs list both directions; keep one per pair
                continue
            seen.add(key)
            err, dur = _num(r, "error"), _num(r, "duration")
            if math.isnan(err) or math.isnan(dur) or err < 0 or err >= MAX_USABLE_ERROR:
                dropped_p += 1
                continue
            _check_seconds("2q gate duration", dur, 1e-10, 1e-5)
            pairs.append(PairCal(err, dur))

        if not qubits or not pairs:
            raise ValueError(f"{name}: no usable calibration rows after filtering")
        return cls(name, qubits, pairs, dropped_q, dropped_p)

    @classmethod
    def from_csv(cls, qubits_csv, pairs_csv, name: str | None = None) -> "DeviceCalibration":
        with open(qubits_csv, newline="") as f:
            q_rows = list(csv.DictReader(f))
        with open(pairs_csv, newline="") as f:
            p_rows = list(csv.DictReader(f))
        return cls.from_rows(q_rows, p_rows, name or Path(qubits_csv).stem)

    def summary(self) -> str:
        e2 = np.median([p.error for p in self.pairs])
        ro = np.median([q.readout_error for q in self.qubits])
        t1 = np.median([q.t1 for q in self.qubits]) * 1e6
        t2 = np.median([q.t2 for q in self.qubits]) * 1e6
        return (
            f"{self.name}: {len(self.qubits)} qubits (-{self.dropped_qubits}), "
            f"{len(self.pairs)} pairs (-{self.dropped_pairs}); median 2q={e2:.2e}, "
            f"readout={ro:.2e}, T1={t1:.0f}us, T2={t2:.0f}us"
        )


# --- noise model ---------------------------------------------------------------
@dataclass
class CalibratedNoise:
    cal: DeviceCalibration
    seed: int = 0
    p_reset: float = 0.0  # preparation/reset flip probability (unknown -> 0)
    t_reset: float = 0.0  # reset duration in seconds (unknown -> 0)
    # Ablation knobs: 1.0 = as calibrated, 0.0 = switch that source off.
    gate1_scale: float = 1.0
    gate2_scale: float = 1.0
    readout_scale: float = 1.0
    idle_scale: float = 1.0
    _q_cache: dict = field(default_factory=dict, repr=False)
    _p_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_csv(cls, qubits_csv, pairs_csv, seed: int = 0, **kw) -> "CalibratedNoise":
        return cls(DeviceCalibration.from_csv(qubits_csv, pairs_csv), seed=seed, **kw)

    @classmethod
    def from_fake(cls, device: str, seed: int = 0, directory=CALIBRATION_DIR, **kw):
        """device in {'boston', 'kingston', 'miami'} (files fake_<device>_*.csv)."""
        d = Path(directory)
        return cls.from_csv(
            d / f"fake_{device}_qubits.csv", d / f"fake_{device}_two_qubit_gates.csv", seed, **kw
        )

    # frozen, stable assignment: same (seed, index) -> same calibration row
    def qubit(self, q: int) -> QubitCal:
        if q not in self._q_cache:
            rng = np.random.default_rng([self.seed, 0, q])
            self._q_cache[q] = self.cal.qubits[int(rng.integers(len(self.cal.qubits)))]
        return self._q_cache[q]

    def pair(self, a: int, b: int) -> PairCal:
        key = (min(a, b), max(a, b))
        if key not in self._p_cache:
            rng = np.random.default_rng([self.seed, 1, *key])
            self._p_cache[key] = self.cal.pairs[int(rng.integers(len(self.cal.pairs)))]
        return self._p_cache[key]

    def idle_channel(self, q: int, t: float):
        """Pauli-twirled thermal relaxation over time t -> (px, py, pz) or None.

        1 - 2(px+py) = exp(-t/T1)   and   1 - 2(py+pz) = exp(-t/T2).
        """
        if t <= 0 or self.idle_scale <= 0:
            return None
        c = self.qubit(q)
        t = t * self.idle_scale
        px = py = (1 - math.exp(-t / c.t1)) / 4
        pz = max((1 - math.exp(-t / c.t2)) / 2 - px, 0.0)
        return (px, py, pz) if px + py + pz > 0 else None

    # ---- main entry point
    def apply(self, ideal: stim.Circuit) -> stim.Circuit:
        """Return a noisy copy of `ideal` (any existing noise is removed first)."""
        circuit = ideal.without_noise().flattened()
        n = circuit.num_qubits
        out = stim.Circuit()
        busy = np.zeros(n)
        layer = 0.0

        def occupy(qs, dur):
            nonlocal layer
            for q in qs:
                busy[q] += dur
            layer = max(layer, max(busy[q] for q in qs) if qs else 0.0)

        def flush():
            nonlocal layer
            if layer > 0:
                for q in range(n):
                    ch = self.idle_channel(q, layer - busy[q])
                    if ch:
                        out.append("PAULI_CHANNEL_1", [q], ch)
            busy[:] = 0.0
            layer = 0.0

        for ins in circuit:
            name = ins.name
            tg = ins.targets_copy()
            if name == "TICK":
                flush()
                out.append(ins)
            elif name in ANNOTATIONS:
                out.append(ins)
            else:
                if not all(t.is_qubit_target for t in tg):
                    raise ValueError(f"{name}: non-qubit targets are not supported")
                qs = [t.value for t in tg]
                if name in VIRTUAL_1Q:
                    out.append(ins)
                elif name in PHYSICAL_1Q:
                    out.append(ins)
                    for q in qs:
                        c = self.qubit(q)
                        p = 1.5 * c.gate1_error * self.gate1_scale
                        if p > 0:
                            out.append("DEPOLARIZE1", [q], min(p, 0.75))
                    for q in qs:
                        occupy([q], self.qubit(q).gate1_duration)
                elif name in TWO_Q:
                    out.append(ins)
                    for a, b in zip(qs[::2], qs[1::2]):
                        c = self.pair(a, b)
                        p = 1.25 * c.error * self.gate2_scale
                        if p > 0:
                            out.append("DEPOLARIZE2", [a, b], min(p, 15 / 16))
                        occupy([a, b], c.duration)
                elif name in MEASURES:
                    flip, with_reset = MEASURES[name]
                    for q in qs:
                        p = self.qubit(q).readout_error * self.readout_scale
                        if p > 0:
                            out.append(flip, [q], p)
                    out.append(ins)
                    if with_reset and self.p_reset > 0:
                        out.append(flip, qs, self.p_reset)
                    for q in qs:
                        occupy([q], self.qubit(q).measure_duration
                               + (self.t_reset if with_reset else 0.0))
                elif name in RESETS:
                    out.append(ins)
                    if self.p_reset > 0:
                        out.append(RESETS[name], qs, self.p_reset)
                    if self.t_reset > 0:
                        occupy(qs, self.t_reset)
                else:
                    raise ValueError(f"Unsupported instruction {name!r}")
        return out
