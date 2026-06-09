"""
Layer 3 — detent spring analysis.

Sweeps detent spring stiffness and measures:
  • Drift of rack from a 50-step displacement under gravity
  • Minimum stiffness where drift < half a tooth pitch

Produces l3_detent_analysis.png.
"""
import numpy as np
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from l3_lane import LaneSimulator, RACK_TOOTH_PITCH

OUTDIR = pathlib.Path(__file__).parent

# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep_stiffness(
    k_values: np.ndarray | None = None,
    n_steps: int = 50,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if k_values is None:
        k_values = np.logspace(0, 3, 30)   # 1 … 1000 N/m

    drifts = []
    for k in k_values:
        sim = LaneSimulator(detent_stiffness=float(k))
        drift = sim.test_detent_holding(n_steps)
        drifts.append(drift)
        if verbose:
            ok = drift < RACK_TOOTH_PITCH / 2
            print(f"  k={k:8.1f} N/m   drift={drift:.4f} mm   "
                  f"[{'HOLD' if ok else 'SLIP'}]")

    return k_values, np.array(drifts)


def find_minimum_stiffness(k_values, drifts) -> float:
    """Return the smallest k where drift < half a tooth pitch."""
    threshold = RACK_TOOTH_PITCH / 2
    idxs = np.where(drifts < threshold)[0]
    if len(idxs) == 0:
        return float("inf")
    return float(k_values[idxs[0]])


def plot_results(k_values, drifts, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    threshold = RACK_TOOTH_PITCH / 2

    ax.semilogx(k_values, drifts, color="#1B4F8A", lw=2, label="Rack drift (mm)")
    ax.axhline(threshold, color="#B85C00", ls="--", lw=1.5,
               label=f"Half-tooth threshold ({threshold:.2f} mm)")

    k_min = find_minimum_stiffness(k_values, drifts)
    if k_min < float("inf"):
        ax.axvline(k_min, color="#1E6B45", ls=":", lw=1.5,
                   label=f"Min stiffness = {k_min:.1f} N/m")

    ax.set_xlabel("Detent spring stiffness k (N/m)", fontsize=10)
    ax.set_ylabel("Rack drift after 50-step displacement (mm)", fontsize=10)
    ax.set_title("Detent Spring Analysis — Minimum Holding Force", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = OUTDIR / "l3_detent_analysis.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close()


def jitter_test(weight: int = 2, n_pulses: int = 16, speed_factor: float = 2.0):
    """Run at 2× normal cam speed and check for double-advance."""
    from l3_lane import CAM_OMEGA, DT
    import pymunk.constraints as PC
    import pymunk

    sim_normal = LaneSimulator(dt=DT)
    _, rack_normal = sim_normal.run_sequence([(weight, n_pulses)])
    final_normal = rack_normal[-1] if rack_normal else 0.0

    sim_fast = LaneSimulator(dt=DT * speed_factor)
    _, rack_fast = sim_fast.run_sequence([(weight, n_pulses)])
    final_fast = rack_fast[-1] if rack_fast else 0.0

    tooth = RACK_TOOTH_PITCH
    double_advance = abs(final_fast - final_normal) > tooth
    print(f"\nJitter test: w={weight:+d}, {n_pulses} pulses at {speed_factor}× speed")
    print(f"  Normal displacement : {final_normal:+.2f} mm")
    print(f"  Fast   displacement : {final_fast:+.2f} mm")
    print(f"  Delta              : {abs(final_fast-final_normal):.2f} mm")
    print(f"  Double-advance?     : {'YES ⚠' if double_advance else 'NO ✓'}")
    return not double_advance


if __name__ == "__main__":
    print("=== Layer 3 — Detent Spring Analysis ===\n")
    k_vals, drifts = sweep_stiffness()
    k_min = find_minimum_stiffness(k_vals, drifts)
    print(f"\nMinimum detent stiffness: {k_min:.1f} N/m")
    print(f"  (half-tooth threshold = {RACK_TOOTH_PITCH/2:.2f} mm)")
    plot_results(k_vals, drifts)
    jitter_test()
