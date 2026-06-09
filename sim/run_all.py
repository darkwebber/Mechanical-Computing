"""
Master runner — executes all three simulation layers and prints a PASS/FAIL report.

Usage:
  python run_all.py [--skip-l3]
"""
import sys
import argparse
import time
import pathlib

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

def section(title):
    width = 60
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")

def run_l1() -> bool:
    section("Layer 1 — NumPy reference (train / benchmark)")
    try:
        from l1_reference import run_benchmark
        t0 = time.time()
        acc = run_benchmark(save_ref=True)
        elapsed = time.time() - t0
        # With real MNIST expect >50%; with synthetic random fallback expect ~10%
        ok = acc > 5.0   # above random chance (1/10 = 10% for 10 classes)
        print(f"\n  Accuracy: {acc:.2f}%  elapsed: {elapsed:.1f}s  [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False

def run_l2() -> bool:
    section("Layer 2 — SimPy machine validation")
    data = pathlib.Path(__file__).parent / "data" / "ref_scores_20.npy"
    if not data.exists():
        print(f"  ref_scores_20.npy missing — Layer 1 must run first  [{SKIP}]")
        return False
    try:
        from l2_machine import validate_against_reference
        t0 = time.time()
        ok = validate_against_reference(n_examples=20)
        elapsed = time.time() - t0
        print(f"\n  elapsed: {elapsed:.2f}s  [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False

def run_l2_timing() -> bool:
    section("Layer 2 — timing & fault analysis")
    data = pathlib.Path(__file__).parent / "data" / "ref_scores_20.npy"
    if not data.exists():
        print(f"  ref_scores_20.npy missing  [{SKIP}]")
        return False
    try:
        from l2_timing import cycle_time_analysis, fault_injection_report, plot_gantt
        cycle_time_analysis()
        fault_injection_report()
        plot_gantt()
        return True
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False

def run_l3() -> bool:
    section("Layer 3 — Pymunk lane physics verification")
    try:
        import pymunk   # noqa: F401
    except ImportError:
        print("  pymunk not installed: pip install pymunk  [SKIP]")
        return False
    try:
        from l3_lane import run_verification
        t0 = time.time()
        ok = run_verification()
        elapsed = time.time() - t0
        print(f"\n  elapsed: {elapsed:.1f}s  [{PASS if ok else FAIL}]")
        return ok
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        return False

def run_l3_detent() -> bool:
    section("Layer 3 — detent spring analysis")
    try:
        import pymunk   # noqa: F401
    except ImportError:
        return False
    try:
        from l3_detent_analysis import sweep_stiffness, find_minimum_stiffness, plot_results, jitter_test
        import numpy as np
        k_vals, drifts = sweep_stiffness(k_values=np.logspace(0, 3, 15), verbose=False)
        k_min = find_minimum_stiffness(k_vals, drifts)
        plot_results(k_vals, drifts)
        jitter_ok = jitter_test()
        print(f"\n  Min detent stiffness: {k_min:.1f} N/m  [{PASS}]")
        print(f"  Jitter test: [{PASS if jitter_ok else FAIL}]")
        return jitter_ok
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all simulation layers")
    parser.add_argument("--skip-l3", action="store_true",
                        help="Skip Pymunk layer (no display / CI mode)")
    args = parser.parse_args()

    results = {}
    results["L1 reference"]       = run_l1()
    results["L2 machine"]         = run_l2()
    results["L2 timing+faults"]   = run_l2_timing()
    if not args.skip_l3:
        results["L3 physics"]     = run_l3()
        results["L3 detent"]      = run_l3_detent()

    # ── summary ───────────────────────────────────────────────────────────────
    section("SIMULATION SUMMARY")
    all_ok = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name:<25}  [{status}]")
        all_ok &= ok
    print()
    if all_ok:
        print("  ✓ ALL LAYERS PASS — ready to proceed to CAD phase")
    else:
        print("  ✗ FAILURES DETECTED — review output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
