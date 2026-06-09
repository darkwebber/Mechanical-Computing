"""
sim/machine_sim.py
Full 10-lane mechanical MNIST classifier — Pymunk physics + Pygame display.

All ten accumulator lanes run in a single Pymunk space.  The machine steps
through the complete classification cycle:
  1. RESET   — all racks snap to centre (tooth 0)
  2. FEATURE — 49 iterations; each advances the 10 racks by weight × score
  3. DONE    — predicted digit (highest rack) highlighted

Run:
  python machine_sim.py [--digit N] [--speed S] [--headless]

Keys:
  SPACE  — pause / resume
  +/-    — double / halve simulation speed
  0-9    — jump to first preset for that digit
  N / P  — next / previous preset
  R      — restart (re-run RESET on the same example)
  Q/ESC  — quit
"""

import math
import sys
import argparse
import pathlib
import numpy as np
import pymunk
import pymunk.constraints as PC

DATA = pathlib.Path(__file__).parent / "data"

# ── physics constants ─────────────────────────────────────────────────────────
_GRAVITY      = 9810.0          # mm/s²
_DT           = 1.0 / 600.0     # physics timestep  s
N_LANES       = 10

# ── machine layout (mm) ───────────────────────────────────────────────────────
LANE_X        = [80.0 + i * 130.0 for i in range(N_LANES)]   # x centres
RACK_Y0       = 300.0           # rest y-position
RACK_TRAVEL   = 150.0           # ±mm display range
RACK_MASS     = 0.01            # kg

# Critically-damped spring (ω ≈ 300 rad/s → fast, smooth snaps)
_SPRING_K     = 900.0           # kg/s²  ≡ N/m  (mm units)
_SPRING_DAMP  = 2.0 * math.sqrt(_SPRING_K * RACK_MASS)

SETTLE_STEPS  = 220             # physics steps between feature advances
RESET_STEPS   = 350

# ── display layout ────────────────────────────────────────────────────────────
WIN_W, WIN_H  = 1400, 800
FEAT_W        = 248             # left panel: feature grid
SCORE_W       = 268             # right panel: scores
LANE_W        = WIN_W - FEAT_W - SCORE_W   # 884 → 88.4 px/lane
LANE_X0       = FEAT_W

RACK_TOP      = 82              # screen y of rack display area top
RACK_BOT      = 700             # screen y of rack display area bottom
RACK_CH       = RACK_BOT - RACK_TOP          # 618 px
RACK_CY       = RACK_TOP + RACK_CH // 2      # zero line y on screen
MM2PX         = RACK_CH / (RACK_TRAVEL * 2)  # px per mm

# colours
BG       = (245, 242, 235)
PAN_BG   = (230, 226, 216)
BLUE     = (27,  79,  138)
BLT      = (80,  125, 195)
INK      = (26,  28,  35)
AMBER    = (200, 90,  10)
GRN      = (22,  115, 60)
GOLD     = (195, 155, 20)
GREY     = (140, 136, 126)
GLTT     = (195, 192, 182)
WHITE    = (255, 255, 255)
TRACK    = (210, 206, 196)

W_COL = {-2: (210,55,10), -1: (215,130,35), 0: (165,161,151), 1: (50,150,75), 2: (20,110,52)}

LANE_COL = [
    (200, 70,  70), (205, 120, 30), (165, 155,  5), (65, 148, 58),
    (28, 128, 115), (28, 88, 172),  (78, 53, 172),  (140, 42, 152),
    (182, 38, 98),  (138, 132, 42),
]


# ── Pymunk lane ───────────────────────────────────────────────────────────────

class _Lane:
    def __init__(self, space: pymunk.Space, idx: int):
        x = LANE_X[idx]

        body = pymunk.Body(RACK_MASS, float("inf"))
        body.position = (x, RACK_Y0)
        seg = pymunk.Segment(body, (0, -90), (0, 90), 3.0)
        seg.filter = pymunk.ShapeFilter(categories=1 << idx, mask=0)
        space.add(body, seg)

        groove = PC.GrooveJoint(
            space.static_body, body,
            groove_a=(x, RACK_Y0 - RACK_TRAVEL - 60),
            groove_b=(x, RACK_Y0 + RACK_TRAVEL + 60),
            anchor_b=(0, 0),
        )
        space.add(groove)

        anchor = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        anchor.position = (x, RACK_Y0)
        space.add(anchor)

        spring = PC.DampedSpring(
            anchor, body, (0, 0), (0, 0), 0.0, _SPRING_K, _SPRING_DAMP
        )
        space.add(spring)

        self.body   = body
        self.anchor = anchor
        self.x      = x

    @property
    def disp(self) -> float:
        return float(self.body.position.y - RACK_Y0)

    def set_target(self, mm: float) -> None:
        mm = max(-RACK_TRAVEL, min(RACK_TRAVEL, mm))
        self.anchor.position = pymunk.Vec2d(self.x, RACK_Y0 + mm)

    def reset(self) -> None:
        self.body.position   = pymunk.Vec2d(self.x, RACK_Y0)
        self.body.velocity   = pymunk.Vec2d(0, 0)
        self.anchor.position = pymunk.Vec2d(self.x, RACK_Y0)


class MachinePhysics:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity  = (0, -_GRAVITY)
        self.space.damping  = 0.985
        self.lanes = [_Lane(self.space, i) for i in range(N_LANES)]

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.space.step(_DT)

    def reset_all(self) -> None:
        for lane in self.lanes:
            lane.reset()

    def set_targets(self, targets_mm) -> None:
        for d, mm in enumerate(targets_mm):
            self.lanes[d].set_target(float(mm))

    def displacements(self) -> list:
        return [lane.disp for lane in self.lanes]


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_model():
    w_p, b_p = DATA / "weights.npy", DATA / "bias.npy"
    if w_p.exists() and b_p.exists():
        return np.load(w_p), np.load(b_p)
    from l1_reference import _generate_synthetic_model
    return _generate_synthetic_model()


def _load_presets():
    ref_p = DATA / "ref_scores_20.npy"
    if ref_p.exists():
        ref = np.load(ref_p, allow_pickle=True).item()
        n = len(ref["labels"])
        return [{"label": int(ref["labels"][i]),
                 "pooled": ref["pooled"][i],
                 "scores": ref["scores"][i]} for i in range(n)]
    from l1_reference import _make_synthetic_test_set, load_model, classify
    W, b = load_model()
    pooled_all, labels = _make_synthetic_test_set(n_per_class=2, seed=7)
    return [{"label": int(labels[i]),
             "pooled": pooled_all[i],
             "scores": classify(pooled_all[i], W, b)}
            for i in range(len(labels))]


def _cumulative_disps(weights, bias, pooled) -> np.ndarray:
    """(50, 10) rack displacements (mm) — row 0 = bias, rows 1-49 = after each feature."""
    cum = np.zeros((50, 10), dtype=np.float64)
    cum[0] = bias.astype(np.float64)
    for f in range(49):
        cum[f + 1] = cum[f] + weights[f].astype(np.float64) * float(pooled[f])
    # Normalise so the maximum final absolute score maps to 80 % of RACK_TRAVEL
    final_abs = np.abs(cum[-1]).max()
    if final_abs > 0:
        scale = RACK_TRAVEL * 0.80 / final_abs
        cum *= scale
    return np.clip(cum, -RACK_TRAVEL, RACK_TRAVEL)


# ── classification state machine ──────────────────────────────────────────────

_IDLE = "IDLE"
_RESET = "RESET"
_RUN = "RUNNING"
_DONE = "DONE"


class ClassSM:
    def __init__(self, physics: MachinePhysics):
        self.phys      = physics
        self.state     = _IDLE
        self.feature   = -1
        self.settle    = 0
        self.weights   = None
        self.bias      = None
        self.pooled    = None
        self.cum_d     = None     # (50,10) mm displacements
        self.label     = -1
        self.predicted = -1
        self.paused    = False
        self.speed     = 10      # physics steps per frame

    def load(self, weights, bias, preset):
        self.weights   = weights
        self.bias      = bias
        self.pooled    = preset["pooled"]
        self.label     = preset["label"]
        self.cum_d     = _cumulative_disps(weights, bias, self.pooled)
        self.predicted = -1
        self.phys.reset_all()
        self.state   = _RESET
        self.feature = -1
        self.settle  = RESET_STEPS

    def restart(self):
        if self.cum_d is not None:
            self.phys.reset_all()
            self.predicted = -1
            self.state   = _RESET
            self.feature = -1
            self.settle  = RESET_STEPS

    def update(self, steps: int):
        if self.paused or self.state in (_IDLE, _DONE):
            return
        rem = steps
        while rem > 0:
            take = min(rem, self.settle)
            self.phys.step(take)
            self.settle -= take
            rem        -= take
            if self.settle <= 0:
                if self.state == _RESET:
                    self.state   = _RUN
                    self.feature = 0
                    self.phys.set_targets(self.cum_d[1])
                    self.settle  = SETTLE_STEPS
                elif self.state == _RUN:
                    self.feature += 1
                    if self.feature >= 49:
                        self.state     = _DONE
                        disps = self.phys.displacements()
                        self.predicted = int(np.argmax(disps))
                        break
                    else:
                        self.phys.set_targets(self.cum_d[self.feature + 1])
                        self.settle = SETTLE_STEPS

    @property
    def progress(self) -> float:
        if self.state == _RESET:
            return 0.0
        if self.state == _DONE:
            return 1.0
        return (self.feature + 1) / 49.0

    @property
    def cur_weights(self):
        if self.weights is None or self.feature < 0:
            return None
        return self.weights[min(self.feature, 48)]


# ── Pygame renderer ───────────────────────────────────────────────────────────

def _font(size, bold=False):
    import pygame
    for name in ("DejaVu Sans Mono", "Consolas", "Courier New", "monospace"):
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            return f
        except Exception:
            continue
    return pygame.font.Font(None, size)


class Renderer:
    def __init__(self, screen):
        import pygame
        self.pg      = pygame
        self.screen  = screen
        self.fn_xl   = _font(18, bold=True)
        self.fn_l    = _font(14, bold=True)
        self.fn_m    = _font(12)
        self.fn_s    = _font(10)

    # helpers
    def _txt(self, text, font, color, x, y):
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def _mm2sy(self, mm):
        return int(RACK_CY - mm * MM2PX)

    def _lane_cx(self, d):
        return int(LANE_X0 + (d + 0.5) * LANE_W / N_LANES)

    # ── feature panel (left) ──────────────────────────────────────────────────

    def _draw_feature_panel(self, sm: ClassSM):
        pg = self.pg
        s  = self.screen
        pg.draw.rect(s, PAN_BG, (0, 0, FEAT_W, WIN_H))
        pg.draw.line(s, GLTT, (FEAT_W - 1, 0), (FEAT_W - 1, WIN_H))

        self._txt("INPUT  IMAGE", self.fn_l, BLUE, 12, 10)

        # 7×7 pooled tile grid
        if sm.pooled is not None:
            cell = 30
            gx, gy = 12, 34
            mx = max(int(sm.pooled.max()), 1)
            for tr in range(7):
                for tc in range(7):
                    idx  = tr * 7 + tc
                    val  = int(sm.pooled[idx])
                    bri  = int(255 * val / mx)
                    col  = (bri, bri, bri)
                    rx   = gx + tc * cell
                    ry   = gy + tr * cell
                    pg.draw.rect(s, col, (rx, ry, cell - 1, cell - 1))
                    if sm.state == _RUN and sm.feature == idx:
                        pg.draw.rect(s, GOLD, (rx, ry, cell - 1, cell - 1), 2)
                    elif sm.state == _RUN and sm.feature > idx:
                        pg.draw.rect(s, (100, 150, 200, 100), (rx, ry, cell - 1, cell - 1), 1)

        # Progress bar
        py = 252
        pg.draw.rect(s, TRACK, (12, py, FEAT_W - 24, 10), border_radius=5)
        pw = int((FEAT_W - 24) * sm.progress)
        if pw > 0:
            pg.draw.rect(s, BLUE, (12, py, pw, 10), border_radius=5)
        fi = max(sm.feature, 0)
        self._txt(f"Feature {fi+1 if sm.state==_RUN else '—'}/49", self.fn_m, INK, 12, py + 14)

        # Weight row for current feature
        wy = 296
        self._txt("Weights  (this feature)", self.fn_s, GREY, 12, wy)
        wy += 16
        if sm.cur_weights is not None:
            bar_max = 50
            for d in range(N_LANES):
                w   = int(sm.cur_weights[d])
                col = W_COL[w]
                # digit chip
                pg.draw.rect(s, LANE_COL[d], (12, wy + d * 24, 20, 18), border_radius=3)
                self._txt(str(d), self.fn_s, WHITE, 18, wy + d * 24 + 3)
                # weight bar
                bw = int(abs(w) / 2 * bar_max)
                if w > 0:
                    pg.draw.rect(s, col, (36, wy + d * 24 + 3, bw, 12), border_radius=2)
                elif w < 0:
                    pg.draw.rect(s, col, (36 + bar_max - bw, wy + d * 24 + 3, bw, 12), border_radius=2)
                # centre line
                pg.draw.line(s, GLTT, (36 + bar_max // 2, wy + d * 24 + 3),
                             (36 + bar_max // 2, wy + d * 24 + 14))
                self._txt(f"{w:+d}", self.fn_s, col, 92, wy + d * 24 + 3)

        # State chip
        st_col = {_IDLE: GREY, _RESET: AMBER, _RUN: BLUE, _DONE: GRN}
        st_y = WIN_H - 80
        pg.draw.rect(s, st_col.get(sm.state, GREY), (12, st_y, FEAT_W - 24, 28), border_radius=5)
        self._txt(sm.state, self.fn_l, WHITE, FEAT_W // 2 - 28, st_y + 5)

        # Digit label
        lbl = f"True label: {sm.label}" if sm.label >= 0 else "—"
        self._txt(lbl, self.fn_m, INK, 12, WIN_H - 46)
        if sm.predicted >= 0:
            pred_col = GRN if sm.predicted == sm.label else AMBER
            self._txt(f"Predicted: {sm.predicted}", self.fn_l, pred_col, 12, WIN_H - 28)

    # ── lane panel (centre) ───────────────────────────────────────────────────

    def _draw_lanes(self, sm: ClassSM):
        pg  = self.pg
        s   = self.screen
        lw  = LANE_W // N_LANES      # px per lane

        # Panel header
        self._txt("ACCUMULATOR  RACKS", self.fn_l, BLUE, LANE_X0 + 8, 10)
        pg.draw.line(s, GLTT, (LANE_X0, 0), (LANE_X0, WIN_H))

        disps = sm.phys.displacements()
        cur_w = sm.cur_weights

        for d in range(N_LANES):
            cx   = self._lane_cx(d)
            disp = disps[d]
            sy   = self._mm2sy(disp)       # screen y of rack centre
            col  = LANE_COL[d]

            # Track (groove)
            track_x = cx - 8
            pg.draw.rect(s, TRACK, (track_x, RACK_TOP, 16, RACK_CH), border_radius=3)

            # Tick marks every 10% travel
            for frac in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                ty = self._mm2sy(frac * RACK_TRAVEL)
                pg.draw.line(s, GLTT, (cx - 16, ty), (cx + 16, ty))
                if frac == 0.0:
                    pg.draw.line(s, GREY, (cx - 16, ty), (cx + 16, ty), 2)

            # Displacement fill bar
            fill_y = min(sy, RACK_CY)
            fill_h = abs(RACK_CY - sy)
            if fill_h > 1:
                bar_col = GRN if disp >= 0 else AMBER
                pg.draw.rect(s, bar_col, (cx - 7, fill_y, 14, fill_h), border_radius=2)

            # Rack body (thick horizontal marker)
            marker_col = col if (sm.state == _DONE and d == sm.predicted) else GLTT
            marker_w   = lw - 10
            pg.draw.rect(s, INK, (cx - marker_w // 2 - 1, sy - 8, marker_w + 2, 16), border_radius=3)
            pg.draw.rect(s, marker_col, (cx - marker_w // 2, sy - 7, marker_w, 14), border_radius=3)

            # Displacement text
            self._txt(f"{disp:+.0f}", self.fn_s, INK, cx - 16, sy - 20)

            # Weight indicator circle
            if cur_w is not None:
                w     = int(cur_w[d])
                wcol  = W_COL[w]
                pg.draw.circle(s, wcol, (cx, RACK_TOP - 18), 10)
                self._txt(f"{w:+d}", self.fn_s, WHITE, cx - 8, RACK_TOP - 24)

            # Digit label
            pg.draw.rect(s, col, (cx - 15, RACK_BOT + 8, 30, 22), border_radius=4)
            self._txt(str(d), self.fn_l, WHITE, cx - 5, RACK_BOT + 10)

            # Predicted star
            if sm.state == _DONE and d == sm.predicted:
                pg.draw.polygon(s, GOLD, _star(cx, RACK_BOT + 42, 12))

    # ── score panel (right) ───────────────────────────────────────────────────

    def _draw_score_panel(self, sm: ClassSM):
        pg = self.pg
        s  = self.screen
        px = LANE_X0 + LANE_W
        pg.draw.rect(s, PAN_BG, (px, 0, SCORE_W, WIN_H))
        pg.draw.line(s, GLTT, (px, 0), (px, WIN_H))

        self._txt("SCORES", self.fn_l, BLUE, px + 12, 10)

        disps = sm.phys.displacements()
        max_d = max(abs(d) for d in disps) if any(disps) else 1.0
        max_d = max(max_d, 1.0)
        bar_max_w = SCORE_W - 80

        for d in range(N_LANES):
            row_y  = 36 + d * 72
            disp   = disps[d]
            is_win = (sm.state == _DONE and d == sm.predicted)
            col    = LANE_COL[d]

            # Background
            if is_win:
                pg.draw.rect(s, (230, 220, 160), (px + 6, row_y, SCORE_W - 12, 66), border_radius=6)
            pg.draw.rect(s, col, (px + 6, row_y, 28, 28), border_radius=4)
            self._txt(str(d), self.fn_xl, WHITE, px + 12, row_y + 3)

            # Digit label
            self._txt(f"Digit  {d}", self.fn_m, INK, px + 40, row_y + 4)

            # Score bar
            bw = int(abs(disp) / max_d * bar_max_w)
            by = row_y + 32
            pg.draw.rect(s, TRACK, (px + 8, by, bar_max_w, 14), border_radius=3)
            bcol = GRN if disp >= 0 else AMBER
            if bw > 0:
                bx = px + 8 if disp >= 0 else px + 8 + bar_max_w - bw
                pg.draw.rect(s, bcol, (bx, by, bw, 14), border_radius=3)
            # centre tick
            midx = px + 8 + bar_max_w // 2
            pg.draw.line(s, GREY, (midx, by), (midx, by + 14))
            self._txt(f"{disp:+5.1f}", self.fn_s, INK, px + 8 + bar_max_w + 4, by)

            # Win badge
            if is_win:
                self._txt("★ PREDICTED", self.fn_s, GOLD, px + 40, row_y + 50)
            if sm.label >= 0 and d == sm.label:
                self._txt("✓ TRUE", self.fn_s, BLT, px + 120 if is_win else px + 40, row_y + 50)

    # ── status bar (bottom) ───────────────────────────────────────────────────

    def _draw_status(self, sm: ClassSM, fps: float):
        pg  = self.pg
        s   = self.screen
        bar_y = WIN_H - 24
        pg.draw.rect(s, BLUE, (0, bar_y, WIN_W, 24))
        ctrl = ("SPACE=pause  +/-=speed  0-9=digit  N/P=preset  R=restart  Q=quit"
                f"   |   speed={sm.speed}×   fps={fps:.0f}")
        self._txt(ctrl, self.fn_s, WHITE, 8, bar_y + 5)

    # ── main draw ─────────────────────────────────────────────────────────────

    def draw(self, sm: ClassSM, fps: float):
        self.screen.fill(BG)
        self._draw_feature_panel(sm)
        self._draw_lanes(sm)
        self._draw_score_panel(sm)
        self._draw_status(sm, fps)
        self.pg.display.flip()


def _star(cx, cy, r):
    """Return 10 vertices for a 5-pointed star centred at (cx,cy) radius r."""
    pts = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        ri    = r if i % 2 == 0 else r * 0.45
        pts.append((cx + ri * math.cos(angle), cy - ri * math.sin(angle)))
    return pts


# ── main ─────────────────────────────────────────────────────────────────────

def run_headless(sm: ClassSM, weights, bias, presets, digit: int):
    """Headless mode: run one classification and print results."""
    idx = next((i for i, p in enumerate(presets) if p["label"] == digit), 0)
    preset = presets[idx]
    print(f"Running classification for digit {preset['label']} (preset {idx}) …")
    sm.load(weights, bias, preset)
    while sm.state != _DONE:
        sm.update(sm.speed)
    disps = sm.phys.displacements()
    print("Final rack displacements (mm):")
    for d, v in enumerate(disps):
        marker = " ← PREDICTED" if d == sm.predicted else ""
        print(f"  digit {d}: {v:+7.2f} mm{marker}")
    correct = "CORRECT" if sm.predicted == sm.label else "WRONG"
    print(f"Predicted: {sm.predicted}  True: {sm.label}  [{correct}]")
    return sm.predicted == sm.label


def main():
    parser = argparse.ArgumentParser(description="Mechanical MNIST classifier simulation")
    parser.add_argument("--digit",    type=int, default=None, help="Starting digit (0-9)")
    parser.add_argument("--speed",    type=int, default=12,   help="Physics steps per frame")
    parser.add_argument("--headless", action="store_true",    help="Run without display")
    args = parser.parse_args()

    weights, bias = _load_model()
    presets       = _load_presets()

    phys  = MachinePhysics()
    sm    = ClassSM(phys)
    sm.speed = args.speed

    if args.headless:
        digit = args.digit if args.digit is not None else 0
        ok = run_headless(sm, weights, bias, presets, digit)
        sys.exit(0 if ok else 1)

    import pygame
    pygame.init()
    screen  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Mechanical MNIST — 10-Lane Classifier")
    clock   = pygame.time.Clock()
    rend    = Renderer(screen)

    # Select starting preset
    start_digit = args.digit if args.digit is not None else 0
    preset_idx  = next((i for i, p in enumerate(presets) if p["label"] == start_digit), 0)
    sm.load(weights, bias, presets[preset_idx])

    running = True
    fps     = 60.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    sm.paused = not sm.paused
                elif k == pygame.K_EQUALS or k == pygame.K_PLUS:
                    sm.speed = min(60, sm.speed * 2)
                elif k == pygame.K_MINUS:
                    sm.speed = max(1, sm.speed // 2)
                elif k == pygame.K_r:
                    sm.restart()
                elif k == pygame.K_n:
                    preset_idx = (preset_idx + 1) % len(presets)
                    sm.load(weights, bias, presets[preset_idx])
                elif k == pygame.K_p:
                    preset_idx = (preset_idx - 1) % len(presets)
                    sm.load(weights, bias, presets[preset_idx])
                elif pygame.K_0 <= k <= pygame.K_9:
                    d = k - pygame.K_0
                    i = next((j for j, p in enumerate(presets) if p["label"] == d), None)
                    if i is not None:
                        preset_idx = i
                        sm.load(weights, bias, presets[preset_idx])

        sm.update(sm.speed)
        rend.draw(sm, fps)
        fps = clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
