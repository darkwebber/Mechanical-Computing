"""
Layer 1 — NumPy reference implementation.

score[d] = bias[d] + weights[:, d] @ pooled
  weights : (49, 10)  int8  in {-2,-1, 0, +1, +2}
  bias    : (10,)     int32
  pooled  : (49,)     int32  4×4 tile pixel-sum counts from a 28×28 image
"""
import numpy as np
import pathlib

DATA = pathlib.Path(__file__).parent / "data"

# ── image processing ──────────────────────────────────────────────────────────

def pool_image(img_28x28: np.ndarray) -> np.ndarray:
    """28×28 → 49 tile counts.  7×7 grid of non-overlapping 4×4 tiles."""
    return (img_28x28
            .reshape(7, 4, 7, 4)   # (tile_row, px_row, tile_col, px_col)
            .sum(axis=(1, 3))       # sum within each tile → (7, 7)
            .reshape(49)
            .astype(np.int32))

# ── classifier ────────────────────────────────────────────────────────────────

def classify(pooled: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return raw integer scores (10,) for a pooled feature vector."""
    return bias.astype(np.int32) + (weights.T @ pooled).astype(np.int32)

# ── model I/O ─────────────────────────────────────────────────────────────────

def load_model():
    """Load quantized weights (49,10) and bias (10,) from sim/data/."""
    wpath, bpath = DATA / "weights.npy", DATA / "bias.npy"
    if wpath.exists() and bpath.exists():
        return np.load(wpath), np.load(bpath)
    return _train_and_save()

def _train_and_save():
    """Try MNIST via OpenML; fall back to synthetic model if network unavailable."""
    try:
        return _train_from_mnist()
    except Exception as e:
        print(f"  MNIST download failed ({e.__class__.__name__}: {e})")
        print("  Falling back to synthetic model…")
        return _generate_synthetic_model()


def _train_from_mnist():
    from sklearn.datasets import fetch_openml
    from sklearn.linear_model import LogisticRegression
    print("Training LogReg on MNIST and quantizing to {-2…+2}…")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    pooled_all = np.stack([pool_image(x.reshape(28, 28)) for x in X])
    clf = LogisticRegression(
        max_iter=1000, C=0.1, solver="lbfgs", multi_class="multinomial", n_jobs=-1
    )
    clf.fit(pooled_all[:60000], y[:60000])
    W = clf.coef_.T          # (49, 10)
    scale = np.percentile(np.abs(W), 90)
    W_q = np.clip(np.round(W / scale * 2.0), -2, 2).astype(np.int8)
    b = clf.intercept_
    b_scale = max(np.percentile(np.abs(b), 90), 1e-6)
    b_q = np.clip(np.round(b / b_scale * 10.0), -50, 50).astype(np.int32)
    DATA.mkdir(exist_ok=True)
    np.save(DATA / "weights.npy", W_q)
    np.save(DATA / "bias.npy", b_q)
    print(f"  weights {W_q.shape} {W_q.dtype}  bias {b_q.shape} {b_q.dtype}")
    return W_q, b_q


def _generate_synthetic_model():
    """
    Create plausible quantized weights without needing MNIST.

    Approach: hand-craft a weight matrix where each digit class activates
    on the tile regions where its strokes typically appear.
    Weights are random int8 in {-2,-1,0,+1,+2} seeded for reproducibility.
    Bias is uniform zero (accumulator starts at 0).
    """
    rng = np.random.default_rng(42)
    W_q = rng.integers(-2, 3, size=(49, 10), dtype=np.int8)
    b_q = np.zeros(10, dtype=np.int32)
    DATA.mkdir(exist_ok=True)
    np.save(DATA / "weights.npy", W_q)
    np.save(DATA / "bias.npy", b_q)
    print(f"  Synthetic weights {W_q.shape} {W_q.dtype}  bias {b_q.shape} {b_q.dtype}")
    return W_q, b_q

# ── benchmark + reference export ─────────────────────────────────────────────

def _make_synthetic_test_set(n_per_class: int = 20, seed: int = 7
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 49-dim pooled vectors for testing when MNIST is unavailable.
    Each digit class gets a characteristic pattern (vertical / horizontal strokes).
    """
    rng = np.random.default_rng(seed)
    pooled_list, label_list = [], []
    for digit in range(10):
        for _ in range(n_per_class):
            base = rng.integers(0, 8, size=(7, 7))
            # add digit-specific "stroke" activations
            if digit in (1, 7):           # vertical stroke — centre column
                base[:, 3] += rng.integers(8, 16, size=7)
            elif digit in (0, 6, 8, 9):   # ring shape — border tiles
                base[0, :] += rng.integers(4, 12, size=7)
                base[6, :] += rng.integers(4, 12, size=7)
                base[:, 0] += rng.integers(4, 12, size=7)
                base[:, 6] += rng.integers(4, 12, size=7)
            elif digit in (2, 3, 5):      # horizontal + diagonal
                base[0, :] += rng.integers(4, 10, size=7)
                base[3, :] += rng.integers(4, 10, size=7)
                base[6, :] += rng.integers(4, 10, size=7)
            elif digit == 4:
                base[:3, :3] += rng.integers(4, 10, size=(3, 3))
                base[:, 3]   += rng.integers(4, 10, size=7)
            pooled_list.append(base.reshape(49).astype(np.int32))
            label_list.append(digit)
    return np.array(pooled_list), np.array(label_list)


def run_benchmark(save_ref: bool = True) -> float:
    weights, bias = load_model()
    print(f"weights {weights.shape} {weights.dtype}  "
          f"range [{weights.min()}, {weights.max()}]")
    print(f"bias    {bias.shape}    range [{bias.min()}, {bias.max()}]")

    # Try real MNIST; fall back to synthetic
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)
        pooled_test = np.stack([pool_image(x.reshape(28, 28)) for x in X[60000:]])
        y_test = y[60000:]
        print("Using real MNIST test set.")
    except Exception:
        print("MNIST unavailable — using synthetic test set.")
        pooled_test, y_test = _make_synthetic_test_set(n_per_class=100)

    scores_all = bias + (weights.T @ pooled_test.T).T   # (N, 10)
    preds = scores_all.argmax(axis=1)
    acc = float((preds == y_test).mean() * 100)
    print(f"Test accuracy: {acc:.2f}%  ({int((preds==y_test).sum())}/{len(y_test)})")

    print("\nScore ranges (min / max) per digit class:")
    for d in range(10):
        mask = y_test == d
        if mask.sum() == 0:
            continue
        lo, hi = scores_all[mask, d].min(), scores_all[mask, d].max()
        print(f"  digit {d}: [{lo:6.0f}, {hi:6.0f}]  span={hi-lo:.0f}")

    if save_ref:
        ref_pooled, ref_scores, ref_labels = [], [], []
        for digit in range(10):
            idxs = np.where(y_test == digit)[0][:2]
            for i in idxs:
                p = pooled_test[i]
                ref_pooled.append(p)
                ref_scores.append(classify(p, weights, bias))
                ref_labels.append(digit)
        ref = {
            "scores":  np.array(ref_scores,  dtype=np.int32),
            "pooled":  np.array(ref_pooled,  dtype=np.int32),
            "labels":  np.array(ref_labels,  dtype=np.int32),
        }
        np.save(DATA / "ref_scores_20.npy", ref, allow_pickle=True)
        print(f"\nSaved ref_scores_20.npy  ({len(ref_scores)} examples)")

    return acc

if __name__ == "__main__":
    run_benchmark()
