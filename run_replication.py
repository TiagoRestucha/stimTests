"""Run the IDE replication (Part 1) + the fingerprinting-vs-distance study.

Usage:
    .venv/bin/python run_replication.py            # full run (~10-20 min, 4 cores)
    .venv/bin/python run_replication.py --quick    # smoke test (~1 min)

Outputs go to results/ (CSV) and figures/ (PNG).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decoder import LogicalResult, collect_logical, detector_firing_rates
from extrapolation import fit_ide, richardson
from fingerprint_analysis import logical_distinguishability, syndrome_shots_needed
from surface_memory import DeviceNoise

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Part A — IDE replication (Fig. 3a analogue)
# --------------------------------------------------------------------------- #
def run_ide(p_values: list[float], fit_ds: list[int], ref_ds: list[int], args: argparse.Namespace) -> None:
    devices = [DeviceNoise(name=f"p={p}", p=p) for p in p_values]
    results = collect_logical(devices, fit_ds + ref_ds, args.max_shots, args.max_errors, args.workers)

    rows, summary = [], []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(devices)))
    d_dense = np.linspace(min(fit_ds), max(ref_ds) + 6, 200)

    for dev, color in zip(devices, colors):
        rs = [r for r in results if r.device == dev.name]
        for r in rs:
            rows.append([dev.p, r.distance, r.shots, r.errors, r.logical_error_rate, r.expectation, r.expectation_std])
        fit_rs = [r for r in rs if r.distance in fit_ds]
        ref_rs = [r for r in rs if r.distance in ref_ds]
        d = np.array([r.distance for r in fit_rs], dtype=float)
        e = np.array([r.expectation for r in fit_rs])
        s = np.array([r.expectation_std for r in fit_rs])

        fit = fit_ide(d, e, s, "single")
        rich = richardson(d, e)
        best_raw = fit_rs[-1].expectation
        summary.append([dev.p, best_raw, 1 - best_raw, fit.extrapolated, abs(1 - fit.extrapolated), fit.extrapolated_std, rich, abs(1 - rich)])

        ax.errorbar(d, e, yerr=s, fmt="o", color=color, label=f"p = {dev.p}")
        if ref_rs:
            ax.errorbar([r.distance for r in ref_rs], [r.expectation for r in ref_rs],
                        yerr=[r.expectation_std for r in ref_rs], fmt="s", color="k", ms=4)
        ax.plot(d_dense, fit.predict(d_dense), "-", color=color, alpha=0.8)
        ax.axhline(fit.extrapolated, ls=":", color=color, alpha=0.6)

    ax.axhline(1.0, color="k", lw=1, label="ideal = 1")
    ax.axvline(max(fit_ds) + 1, color="grey", lw=1)
    ax.set_xlabel("Surface code distance d")
    ax.set_ylabel(r"$\langle 0|Z_L|0\rangle$")
    ax.set_title("IDE: fit $A - Be^{-Cd}$ on coloured points, black = held-out")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "ide_expectation_vs_distance.png", dpi=150)

    _write_csv(RESULTS / "ide_raw.csv", ["p", "d", "shots", "errors", "p_L", "E", "E_std"], rows)
    _write_csv(RESULTS / "ide_summary.csv",
               ["p", f"E_raw(d={max(fit_ds)})", "err_raw", "E_IDE", "err_IDE", "E_IDE_std", "E_Richardson", "err_Richardson"],
               summary)
    print("\n=== IDE summary (ideal = 1) ===")
    print(f"{'p':>8} {'err_raw':>10} {'err_IDE':>10} {'err_Rich':>10}")
    for row in summary:
        print(f"{row[0]:>8} {row[2]:>10.2e} {row[4]:>10.2e} {row[7]:>10.2e}")


# --------------------------------------------------------------------------- #
# Part B — fingerprinting vs distance
# --------------------------------------------------------------------------- #
def run_fingerprint(p: float, ds: list[int], args: argparse.Namespace) -> None:
    chip_1 = DeviceNoise("chip_1", p, inhomogeneity=0.3, seed=1)
    chip_2 = DeviceNoise("chip_2", p, inhomogeneity=0.3, seed=2)
    results = collect_logical([chip_1, chip_2], ds, args.max_shots, args.max_errors, args.workers)
    res_1 = [r for r in results if r.device == chip_1.name]
    res_2 = [r for r in results if r.device == chip_2.name]
    logical = logical_distinguishability(res_1, res_2)

    syndrome_n = []
    for d in ds:
        ra = detector_firing_rates(chip_1, d, args.syndrome_shots, seed=11)
        rb = detector_firing_rates(chip_2, d, args.syndrome_shots, seed=22)
        syndrome_n.append(syndrome_shots_needed(ra, rb))

    rows = [[x.distance, x.p_a, x.p_b, x.tvd, x.ratio, x.shots_needed, sn] for x, sn in zip(logical, syndrome_n)]
    _write_csv(RESULTS / "fingerprint.csv",
               ["d", "pL_chip1", "pL_chip2", "TVD", "ratio", "shots_needed_logical", "shots_needed_syndrome"], rows)

    print(f"\n=== Fingerprinting chip_1 vs chip_2 (same spec p={p}, sigma=0.3) ===")
    print(f"{'d':>3} {'pL_1':>10} {'pL_2':>10} {'TVD':>10} {'N*_logical':>12} {'N*_syndrome':>12}")
    for row in rows:
        print(f"{row[0]:>3} {row[1]:>10.2e} {row[2]:>10.2e} {row[3]:>10.2e} {row[5]:>12.3g} {row[6]:>12.3g}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    d_arr = [x.distance for x in logical]
    ax1.semilogy(d_arr, [x.p_a for x in logical], "o-", label="chip 1 $p_L$")
    ax1.semilogy(d_arr, [x.p_b for x in logical], "s-", label="chip 2 $p_L$")
    ax1.semilogy(d_arr, [x.tvd for x in logical], "k^--", label="TVD (logical)")
    ax1.set_xlabel("Code distance d")
    ax1.set_title("Logical signal available for fingerprinting")
    ax1.legend()
    ax2.semilogy(d_arr, [x.shots_needed for x in logical], "o-", label="logical outcomes only")
    ax2.semilogy(ds, syndrome_n, "s-", label="raw syndrome (detectors)")
    ax2.set_xlabel("Code distance d")
    ax2.set_ylabel("Shots to distinguish chips")
    ax2.set_title("Cost of fingerprinting vs distance")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fingerprint_vs_distance.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.quick:
        args.max_shots, args.max_errors, args.syndrome_shots = 50_000, 1_000, 5_000
        p_values, fit_ds, ref_ds = [0.003, 0.004], [3, 5, 7], [9]
    else:
        args.max_shots, args.max_errors, args.syndrome_shots = 2_000_000, 20_000, 50_000
        p_values, fit_ds, ref_ds = [0.002, 0.003, 0.004, 0.005], [3, 5, 7, 9], [11]

    run_ide(p_values, fit_ds, ref_ds, args)
    run_fingerprint(0.003, fit_ds + ref_ds, args)
    print(f"\nCSV -> {RESULTS}\nPNG -> {FIGURES}")


if __name__ == "__main__":
    main()
