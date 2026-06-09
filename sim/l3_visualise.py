"""
Layer 3 — Pygame real-time visualisation of the single lane.

Run:  python l3_visualise.py [--headless]

Keys:
  SPACE  — pause / resume
  +/-    — speed up / slow down (change time_scale)
  Q/ESC  — quit
"""
import math
import sys
import argparse
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame

from l3_lane import (
    LaneSimulator, _make_cam, _make_follower, _make_rack_with_stiffness,
    CAM_CENTER, CAM_BASE_RADIUS, LOBE_HEIGHTS,
    FOLLOWER_PIVOT, FOLLOWER_LEN,
    RACK_X, RACK_Y0, RACK_TOOTH_PITCH,
    DT, CAM_OMEGA,
)
import pymunk.constraints as PC

# ── display constants ─────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 700, 600
SIM_ORIGIN      = (0, HEIGHT)   # Pymunk +y up; flip for pygame
FPS_CAP         = 60
BG_COLOR        = (245, 242, 235)   # blueprint cream
INK             = (26,  28,  35)
BLUE            = (27,  79, 138)
AMBER           = (184, 92,  0)
GREEN           = (30, 107, 69)
GREY            = (107, 104, 96)

WEIGHT_COLORS   = {-2: AMBER, -1: (220,140,0), 0: GREY, 1: (60,150,80), 2: GREEN}
WEIGHT_LABELS   = {-2:"SUB2", -1:"SUB1", 0:"IGN", 1:"ADD1", 2:"ADD2"}

# ── demo instruction sequence ─────────────────────────────────────────────────
DEMO_SEQUENCE = [
    (+2, 8),
    (+1, 12),
    ( 0, 5),
    (-1, 6),
    (-2, 4),
    (+2, 10),
]


class LaneVis:
    def __init__(self, headless: bool = False):
        self.headless = headless
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Mechanical MNIST — Lane Physics (Layer 3)")
            self.clock  = pygame.time.Clock()
            self.font_s = pygame.font.SysFont("monospace", 11)
            self.font_m = pygame.font.SysFont("monospace", 13)

        self.time_scale = 1.0   # multiplier on DT
        self.paused     = False
        self.t          = 0.0
        self.rack_steps = 0     # integer step count
        self.history    = []    # (t, rack_y) for HUD plot

        self._init_sequence()

    def _init_sequence(self):
        self.seq_idx     = 0
        self.seq_pulse   = 0
        self.sequence    = DEMO_SEQUENCE
        w, n = self.sequence[0]
        self._build_space(w)
        self.current_w = w
        self.pulses_left = n
        self.seq_total_expected = sum(abs(w)*n for w, n in self.sequence)

    def _build_space(self, weight_state: int):
        self.space = pymunk.Space()
        self.space.gravity = (0, -9810)
        self.space.damping = 0.95
        self.cam_body, self.cam_shape = _make_cam(self.space, weight_state)
        self.follower_body, self.follower_seg = _make_follower(self.space)
        self.rack_body = _make_rack_with_stiffness(self.space, 10.0)
        if not self.headless:
            self.draw_opts = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_opts.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES

    def _step(self):
        if self.paused:
            return
        steps_per_frame = max(1, int(self.time_scale))
        for _ in range(steps_per_frame):
            self.space.step(DT)
            self.t += DT

        rack_y = self.rack_body.position.y - RACK_Y0
        self.history.append((self.t, rack_y))
        if len(self.history) > 400:
            self.history.pop(0)

        # advance demo sequence
        self._advance_sequence()

    def _advance_sequence(self):
        if self.seq_idx >= len(self.sequence):
            return
        # after ~0.3s of sim time per pulse, emit a "pulse"
        pulse_period = 0.3
        if self.t >= (self.seq_pulse + 1) * pulse_period:
            self.seq_pulse += 1
            self.pulses_left -= 1
            if self.pulses_left <= 0:
                self.seq_idx += 1
                if self.seq_idx < len(self.sequence):
                    w, n = self.sequence[self.seq_idx]
                    self.current_w  = w
                    self.pulses_left = n
                    self._build_space(w)

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_EQUALS:
                    self.time_scale = min(16.0, self.time_scale * 2)
                if event.key == pygame.K_MINUS:
                    self.time_scale = max(0.25, self.time_scale / 2)
        return True

    def _draw_hud(self):
        s = self.screen
        # title
        t = self.font_m.render("Mechanical MNIST — Lane Physics Simulation", True, BLUE)
        s.blit(t, (10, 10))

        # weight state indicator
        w = self.current_w
        col = WEIGHT_COLORS[w]
        label = WEIGHT_LABELS[w]
        pygame.draw.rect(s, col, (10, 32, 80, 22), border_radius=3)
        lbl = self.font_m.render(label, True, (255,255,255))
        s.blit(lbl, (14, 35))

        seq_lbl = self.font_s.render(
            f"Step {self.seq_idx+1}/{len(self.sequence)}  "
            f"pulses_left={self.pulses_left}  "
            f"t={self.t:.2f}s  speed={self.time_scale:.1f}×",
            True, GREY)
        s.blit(seq_lbl, (10, 58))

        # rack displacement bar (right side)
        rack_y = self.rack_body.position.y - RACK_Y0
        bar_x, bar_y0, bar_h = WIDTH - 60, HEIGHT // 2, 200
        pygame.draw.rect(s, (210,206,196), (bar_x, bar_y0 - bar_h//2, 30, bar_h), 1)
        fill_h = int(np.clip(rack_y / (50 * RACK_TOOTH_PITCH) * bar_h/2, -bar_h//2, bar_h//2))
        col2 = GREEN if rack_y >= 0 else AMBER
        if fill_h > 0:
            pygame.draw.rect(s, col2, (bar_x, bar_y0 - fill_h, 30, fill_h))
        elif fill_h < 0:
            pygame.draw.rect(s, col2, (bar_x, bar_y0, 30, -fill_h))
        disp_lbl = self.font_s.render(f"{rack_y:+.1f}mm", True, INK)
        s.blit(disp_lbl, (bar_x - 5, bar_y0 + bar_h//2 + 4))
        acc_lbl = self.font_s.render("ACC", True, BLUE)
        s.blit(acc_lbl, (bar_x + 4, bar_y0 - bar_h//2 - 16))

        # mini plot of rack history
        if len(self.history) > 1:
            pts = [(10 + int(i * (WIDTH - 140) / 400),
                    HEIGHT - 20 - int(np.clip(ry / (100*RACK_TOOTH_PITCH), -1, 1) * 40))
                   for i, (_, ry) in enumerate(self.history)]
            pygame.draw.lines(s, BLUE, False, pts, 1)

        # controls legend
        ctrl = self.font_s.render("SPACE=pause  +/-=speed  Q=quit", True, GREY)
        s.blit(ctrl, (10, HEIGHT - 18))

    def run(self, max_seconds: float = 20.0) -> list:
        """Run the visualisation loop. Returns rack position history."""
        if self.headless:
            while self.t < max_seconds:
                self._step()
            return self.history

        running = True
        while running and self.t < max_seconds:
            running = self._handle_events()
            self.screen.fill(BG_COLOR)
            self.space.debug_draw(self.draw_opts)
            self._draw_hud()
            self._step()
            pygame.display.flip()
            self.clock.tick(FPS_CAP)

        pygame.quit()
        return self.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (CI / headless server)")
    args = parser.parse_args()
    vis = LaneVis(headless=args.headless)
    history = vis.run()
    print(f"Ran {len(history)} frames, final rack_y={history[-1][1]:+.2f}mm")


if __name__ == "__main__":
    main()
