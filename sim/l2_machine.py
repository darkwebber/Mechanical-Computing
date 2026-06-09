"""
Layer 2 — SimPy discrete-event machine simulation.

Models the mechanical classifier as timed SimPy processes:
  • ResetCam  — fires 9 cam events that reset all 10 accumulators to bias
  • PulseReplay — replays N solenoid pulses per feature cycle
  • WeightLane  — per-digit process: reads weight, updates accumulator level
  • FeatureSeq  — advances row pointer after each feature cycle completes

Final accumulator levels must match the Layer-1 reference scores.
"""
import simpy
import numpy as np
import pathlib
from typing import Dict, List, Tuple

DATA = pathlib.Path(__file__).parent / "data"

# ── timing constants (arbitrary but proportional to real machine) ─────────────
T_PULSE          = 1.0    # one solenoid pulse period
T_ADVANCE        = 0.5    # sequencer cam advances to next row
T_RESET_CAM_STEP = 1.0    # time between consecutive reset-cam lobes
T_RESET_SETTLE   = 2.0    # extra settle time after last reset event

# Reset cam lobe schedule: (angular_time, [accumulator_indices_reset])
# 9 cam events cover 10 accumulators (event 0 handles digits 0+1 together)
RESET_CAM_SCHEDULE: List[Tuple[float, List[int]]] = [
    (0.0, [0, 1]),
    (1.0, [2]),
    (2.0, [3]),
    (3.0, [4]),
    (4.0, [5]),
    (5.0, [6]),
    (6.0, [7]),
    (7.0, [8]),
    (8.0, [9]),
]

# ── event log entry ───────────────────────────────────────────────────────────

class Event:
    __slots__ = ("t", "kind", "data")
    def __init__(self, t, kind, **data):
        self.t, self.kind, self.data = t, kind, data
    def __repr__(self):
        return f"[{self.t:8.2f}] {self.kind}  {self.data}"

# ── machine model ─────────────────────────────────────────────────────────────

class MNISTMachine:
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        """
        weights : (49, 10) int8  quantized to {-2,-1,0,+1,+2}
        bias    : (10,)    int32
        """
        self.weights  = weights
        self.bias     = bias
        self.n_digits = weights.shape[1]   # 10

    # ── public entry point ────────────────────────────────────────────────────

    def classify(self, pooled: np.ndarray) -> Tuple[np.ndarray, List[Event], float]:
        """
        Run a full classify cycle for one image.

        Returns
        -------
        acc_final : (10,) int32  final accumulator levels
        log       : list[Event]  complete event log
        cycle_time: float        total simulated time
        """
        env   = simpy.Environment()
        state = _MachineState(env, self.weights, self.bias)
        env.process(state.run(pooled))
        env.run()
        return state.accumulators.copy(), state.log, env.now

    # ── fault injection variants ──────────────────────────────────────────────

    def classify_with_double_advance(self, pooled: np.ndarray, fault_at_feature: int = 10):
        """Inject a spurious extra advance at feature fault_at_feature."""
        env   = simpy.Environment()
        state = _MachineState(env, self.weights, self.bias, double_advance_at=fault_at_feature)
        env.process(state.run(pooled))
        env.run()
        return state.accumulators.copy(), state.log, env.now

    def classify_with_extra_pulse(self, pooled: np.ndarray, fault_at_feature: int = 5):
        """Inject one extra pulse at feature fault_at_feature."""
        env   = simpy.Environment()
        state = _MachineState(env, self.weights, self.bias, extra_pulse_at=fault_at_feature)
        env.process(state.run(pooled))
        env.run()
        return state.accumulators.copy(), state.log, env.now


class _MachineState:
    def __init__(self, env, weights, bias,
                 double_advance_at=None, extra_pulse_at=None):
        self.env           = env
        self.weights       = weights
        self.bias          = bias
        self.accumulators  = np.array(bias, dtype=np.int32)
        self.log: List[Event] = []
        self.double_advance_at = double_advance_at
        self.extra_pulse_at    = extra_pulse_at

    def _evt(self, kind, **data):
        self.log.append(Event(self.env.now, kind, **data))

    # ── reset sequence ────────────────────────────────────────────────────────

    def _reset(self):
        env = self.env
        self._evt("RESET_START")
        t_cursor = 0.0
        for cam_t, acc_list in RESET_CAM_SCHEDULE:
            yield env.timeout(cam_t - t_cursor)
            t_cursor = cam_t
            for a in acc_list:
                self.accumulators[a] = int(self.bias[a])
                self._evt("CAM_RESET", acc=a, value=int(self.bias[a]))
        yield env.timeout(T_RESET_SETTLE)
        self._evt("RESET_DONE")

    # ── feature cycle ─────────────────────────────────────────────────────────

    def _feature_cycle(self, feature: int, count: int):
        env = self.env
        # optionally inject extra pulse
        n_pulses = count + (1 if feature == self.extra_pulse_at else 0)
        self._evt("FEATURE_START", f=feature, count=count, n_pulses=n_pulses)

        for _ in range(n_pulses):
            yield env.timeout(T_PULSE)
            # All 10 lanes update simultaneously on each pulse
            for d in range(self.weights.shape[1]):
                w = int(self.weights[feature, d])
                self.accumulators[d] += w
            self._evt("PULSE", f=feature,
                      acc=list(self.accumulators.astype(int)))

        yield env.timeout(T_ADVANCE)
        self._evt("FEATURE_END", f=feature)

        # optionally inject a spurious extra advance (skips a feature row)
        if feature == self.double_advance_at:
            self._evt("FAULT_DOUBLE_ADVANCE", f=feature)
            yield env.timeout(T_ADVANCE)

    # ── master run ────────────────────────────────────────────────────────────

    def run(self, pooled: np.ndarray):
        yield from self._reset()
        for f in range(49):
            count = int(pooled[f])
            yield from self._feature_cycle(f, count)
        self._evt("CLASSIFY_DONE",
                  prediction=int(np.argmax(self.accumulators)),
                  scores=list(self.accumulators.astype(int)))

# ── validation against Layer-1 oracle ────────────────────────────────────────

def validate_against_reference(n_examples: int = 20) -> bool:
    """
    Load ref_scores_20.npy produced by Layer 1 and assert that the SimPy
    machine produces identical scores for every example.
    """
    ref_path = DATA / "ref_scores_20.npy"
    if not ref_path.exists():
        raise FileNotFoundError(
            "ref_scores_20.npy not found — run l1_reference.py first"
        )
    ref = np.load(ref_path, allow_pickle=True).item()
    weights = np.load(DATA / "weights.npy")
    bias    = np.load(DATA / "bias.npy")

    machine = MNISTMachine(weights, bias)
    all_pass = True
    for i in range(min(n_examples, len(ref["pooled"]))):
        pooled = ref["pooled"][i]
        expected = ref["scores"][i]
        label    = ref["labels"][i]
        got, _, _ = machine.classify(pooled)
        ok = np.array_equal(got, expected)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] example {i:2d}  digit={label}  "
              f"pred={int(np.argmax(got))}  "
              f"L1={list(expected)}  L2={list(got)}")
    return all_pass

if __name__ == "__main__":
    print("=== Layer 2 — machine validation ===\n")
    ok = validate_against_reference()
    print(f"\nResult: {'ALL PASS' if ok else 'FAILURES DETECTED'}")
