"""
Layer 2 — timing analysis and fault injection.

Produces:
  • Gantt chart of events for one classify cycle
  • Per-feature cycle-time breakdown
  • Fault injection report (double-advance, extra-pulse)
"""
import numpy as np
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from l2_machine import MNISTMachine, T_PULSE, T_ADVANCE

DATA    = pathlib.Path(__file__).parent / "data"
OUTDIR  = pathlib.Path(__file__).parent

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_example(idx: int = 0):
    ref = np.load(DATA / "ref_scores_20.npy", allow_pickle=True).item()
    return ref["pooled"][idx], ref["labels"][idx], ref["scores"][idx]

def _build_machine():
    return MNISTMachine(
        np.load(DATA / "weights.npy"),
        np.load(DATA / "bias.npy"),
    )

# ── Gantt chart ───────────────────────────────────────────────────────────────

def plot_gantt(example_idx: int = 0, save: bool = True):
    pooled, label, _ = _load_example(example_idx)
    machine = _build_machine()
    _, log, total = machine.classify(pooled)

    fig, ax = plt.subplots(figsize=(14, 5))
    color_map = {
        "RESET_START": None, "RESET_DONE": None, "CAM_RESET": "#2E6EB5",
        "FEATURE_START": None, "FEATURE_END": None,
        "PULSE": "#1E6B45",
        "CLASSIFY_DONE": None,
    }
    # Collect feature spans
    f_starts: dict = {}
    for evt in log:
        if evt.kind == "FEATURE_START":
            f_starts[evt.data["f"]] = evt.t
        elif evt.kind == "FEATURE_END":
            f = evt.data["f"]
            t0 = f_starts.get(f, evt.t)
            count = next(e.data["count"] for e in log
                         if e.kind == "FEATURE_START" and e.data["f"] == f)
            ax.barh(0, evt.t - t0, left=t0, height=0.4,
                    color="#EAF0F9", edgecolor="#1B4F8A", linewidth=0.4)
            if count > 0:
                ax.text(t0 + (evt.t - t0) / 2, 0,
                        f"f{f}\n×{count}", ha="center", va="center",
                        fontsize=5.5, color="#1B4F8A")

    # Reset bar
    reset_end = next(e.t for e in log if e.kind == "RESET_DONE")
    ax.barh(0, reset_end, left=0, height=0.4, color="#FDF3E7",
            edgecolor="#B85C00", linewidth=0.8)
    ax.text(reset_end / 2, 0, "RESET", ha="center", va="center",
            fontsize=7, color="#B85C00", fontweight="bold")

    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Simulated time (T_PULSE units)", fontsize=9)
    ax.set_title(f"SimPy event timeline — digit {label}  (total={total:.1f})", fontsize=10)

    patches = [
        mpatches.Patch(color="#FDF3E7", label="Reset phase"),
        mpatches.Patch(color="#EAF0F9", label="Feature cycle (count×pulse width)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    plt.tight_layout()
    if save:
        out = OUTDIR / "l2_gantt.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()
    plt.close()

# ── cycle time breakdown ──────────────────────────────────────────────────────

def cycle_time_analysis():
    pooled, label, _ = _load_example(0)
    machine = _build_machine()
    _, log, total = machine.classify(pooled)

    f_starts: dict = {}
    cycle_times = []
    for evt in log:
        if evt.kind == "FEATURE_START":
            f_starts[evt.data["f"]] = evt.t
        elif evt.kind == "FEATURE_END":
            f = evt.data["f"]
            dt = evt.t - f_starts[f]
            count = next(e.data["count"] for e in log
                         if e.kind == "FEATURE_START" and e.data["f"] == f)
            cycle_times.append((f, count, dt))

    print(f"\nCycle time analysis for digit {label}:")
    print(f"  {'Feature':>7}  {'count':>6}  {'cycle_t':>8}  predicted={count*T_PULSE+T_ADVANCE:.1f}")
    total_pulse_t = 0
    for f, cnt, dt in cycle_times:
        total_pulse_t += cnt * T_PULSE
        expected = cnt * T_PULSE + T_ADVANCE
        status = "OK" if abs(dt - expected) < 1e-9 else "ERR"
        if cnt > 0:
            print(f"  f={f:3d}  cnt={cnt:4d}  t={dt:7.1f}  [{status}]")

    reset_t = next(e.t for e in log if e.kind == "RESET_DONE")
    print(f"\n  Reset time     : {reset_t:.1f}")
    print(f"  Total pulse t  : {total_pulse_t:.1f}")
    print(f"  Total advance t: {49*T_ADVANCE:.1f}")
    print(f"  Grand total    : {total:.1f}")
    return cycle_times

# ── fault injection ───────────────────────────────────────────────────────────

def fault_injection_report():
    pooled, label, expected = _load_example(0)
    machine = _build_machine()

    normal, _, _ = machine.classify(pooled)

    print(f"\nFault injection report for digit {label}:")
    print(f"  Normal prediction: {int(np.argmax(normal))} (correct={label})")

    # Double-advance at feature 10
    da_scores, da_log, _ = machine.classify_with_double_advance(pooled, fault_at_feature=10)
    delta_da = da_scores - normal
    print(f"\n  Fault: double-advance at f=10")
    print(f"  Prediction changed: {int(np.argmax(normal))} → {int(np.argmax(da_scores))}")
    print(f"  Score deltas: {list(delta_da)}")

    # Extra pulse at feature 5
    ep_scores, ep_log, _ = machine.classify_with_extra_pulse(pooled, fault_at_feature=5)
    delta_ep = ep_scores - normal
    print(f"\n  Fault: extra pulse at f=5")
    print(f"  Prediction changed: {int(np.argmax(normal))} → {int(np.argmax(ep_scores))}")
    print(f"  Score deltas: {list(delta_ep)}")

    return delta_da, delta_ep

if __name__ == "__main__":
    print("=== Layer 2 — timing & fault analysis ===")
    cycle_time_analysis()
    fault_injection_report()
    plot_gantt()
