"""Add realistic noise to ANY ideal Stim circuit without touching its structure.

Stim has no `with_noise()` helper: noise is just instructions in the circuit
(X_ERROR, DEPOLARIZE1/2, PAULI_CHANNEL_1/2, CORRELATED_ERROR, ...).
`apply_noise` walks an ideal circuit and inserts them after gates, before
measurements, after resets and on every TICK (idle T1/T2 noise).
Rates can be per-qubit / per-pair, e.g. taken from an IBM fake backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pymatching
import stim

TWO_Q = {"CX", "CZ", "CY", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SWAP", "ISWAP"}
ONE_Q = {"H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "I"}
RESET_Z, RESET_X = {"R"}, {"RX"}
MEAS_Z, MEAS_X = {"M", "MR"}, {"MX", "MRX"}
HAS_RESET = {"MR", "MRX"}


@dataclass
class NoiseModel:
    p1: float = 0.0  # 1q gate depolarizing
    p2: float = 0.0  # default 2q gate depolarizing
    p_meas: float = 0.0  # default readout flip
    p_reset: float = 0.0  # default reset flip
    p2_pair: dict = field(default_factory=dict)  # {(a, b): p} overrides
    p_meas_q: dict = field(default_factory=dict)  # {q: p} overrides
    t1: dict = field(default_factory=dict)  # {q: seconds}
    t2: dict = field(default_factory=dict)  # {q: seconds}
    idle_time: float = 0.0  # seconds of idling per TICK layer

    def pair(self, a, b):
        return self.p2_pair.get((a, b), self.p2_pair.get((b, a), self.p2))

    def idle_channel(self, q):
        """Pauli-twirled thermal relaxation -> (px, py, pz)."""
        t1, t2, t = self.t1.get(q), self.t2.get(q), self.idle_time
        if not (t1 and t2 and t):
            return None
        px = py = (1 - np.exp(-t / t1)) / 4
        pz = max((1 - np.exp(-t / t2)) / 2 - px, 0.0)
        return px, py, pz


def apply_noise(ideal: stim.Circuit, m: NoiseModel) -> stim.Circuit:
    ideal = ideal.without_noise().flattened()
    n = ideal.num_qubits
    out = stim.Circuit()
    for ins in ideal:
        name, tg = ins.name, ins.targets_copy()
        qs = [t.value for t in tg if t.is_qubit_target]
        if name in MEAS_Z | MEAS_X:
            for q in qs:  # flip BEFORE measuring (X-basis readout flips with Z)
                p = m.p_meas_q.get(q, m.p_meas)
                if p:
                    out.append("Z_ERROR" if name in MEAS_X else "X_ERROR", [q], p)
            out.append(ins)
            if name in HAS_RESET and m.p_reset:
                out.append("Z_ERROR" if name in MEAS_X else "X_ERROR", qs, m.p_reset)
        elif name in RESET_Z | RESET_X:
            out.append(ins)
            if m.p_reset:
                out.append("Z_ERROR" if name in RESET_X else "X_ERROR", qs, m.p_reset)
        elif name in TWO_Q:
            out.append(ins)
            for a, b in zip(qs[::2], qs[1::2]):
                p = m.pair(a, b)
                if p:
                    out.append("DEPOLARIZE2", [a, b], p)
        elif name in ONE_Q:
            out.append(ins)
            if m.p1:
                out.append("DEPOLARIZE1", qs, m.p1)
        elif name == "TICK":
            for q in range(n):
                ch = m.idle_channel(q)
                if ch:
                    out.append("PAULI_CHANNEL_1", [q], ch)
            out.append(ins)
        else:
            out.append(ins)
    return out


def logical_error_rate(circuit: stim.Circuit, shots: int = 200_000) -> float:
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = circuit.compile_detector_sampler().sample(
        shots, separate_observables=True
    )
    return float(np.mean((obs ^ matching.decode_batch(dets))[:, 0]))


# ----------------------------------------------------------------- IBM backend
def best_chain(backend, length, gate="cz"):
    """Best linear chain of `length` physical qubits (lowest 2q+readout error)."""
    t = backend.target
    gate = gate if gate in t else "cx"
    err2 = {k: v.error for k, v in t[gate].items() if v and v.error is not None}
    adj = {}
    for (a, b), e in err2.items():
        adj.setdefault(a, {})[b] = e
    meas = {k[0]: v.error for k, v in t["measure"].items() if v and v.error is not None}

    def extend(path):
        if len(path) == length:
            yield path
            return
        for nb in adj.get(path[-1], {}):
            if nb not in path:
                yield from extend(path + [nb])

    def score(p):
        e2 = sum(adj[a][b] for a, b in zip(p, p[1:]))
        return e2 + sum(meas.get(q, 1) for q in p[1::2])  # ancilla readouts

    return min((p for s in adj for p in extend([s])), key=score)


def model_from_backend(backend, chain, gate="cz", idle_time=None) -> NoiseModel:
    """Per-qubit/per-pair NoiseModel; circuit qubit i <-> physical chain[i]."""
    t = backend.target
    gate = gate if gate in t else "cx"
    m = NoiseModel()
    for i, j in zip(range(len(chain)), range(1, len(chain))):
        p = t[gate][(chain[i], chain[j])]
        m.p2_pair[(i, j)] = p.error
        if idle_time is None and p.duration:
            idle_time = p.duration
    for i, q in enumerate(chain):
        m.p_meas_q[i] = t["measure"][(q,)].error
        qp = backend.qubit_properties(q)
        m.t1[i], m.t2[i] = qp.t1, qp.t2
    m.p_reset = float(np.median(list(m.p_meas_q.values()))) / 10  # rough guess
    m.idle_time = idle_time or 0.0
    return m


if __name__ == "__main__":
    from qiskit_ibm_runtime.fake_provider import (
        FakeBoston,
        FakeKingston,
        FakeMiami,
    )

    d, rounds = 3, 3
    # IDEAL circuit: your repetition code with all noise removed.
    ideal = stim.Circuit.generated(
        "repetition_code:memory", distance=d, rounds=rounds
    )
    n = ideal.num_qubits
    print(f"{'backend':14}{'2q only':>10}{'meas only':>11}{'idle only':>11}{'all':>10}")
    for B in [ FakeBoston, FakeKingston, FakeMiami]:
        b = B()
        gate = "cz" if "cz" in b.target else "cx"
        chain = best_chain(b, n, gate)
        full = model_from_backend(b, chain, gate)
        only2 = NoiseModel(p2_pair=full.p2_pair)
        onlym = NoiseModel(p_meas_q=full.p_meas_q)
        onlyi = NoiseModel(t1=full.t1, t2=full.t2, idle_time=full.idle_time)
        rates = [
            logical_error_rate(apply_noise(ideal, mm))
            for mm in (only2, onlym, onlyi, full)
        ]
        print(f"{b.name:14}" + "".join(f"{r:>11.6f}" for r in rates), chain)
