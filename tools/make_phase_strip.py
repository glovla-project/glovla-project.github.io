#!/usr/bin/env python3
"""Phase filmstrip: one panel per rollout phase with a phase-colored gripper trace.

Panels follow the HybridVLA state machine:
  reset -> planner->object -> grasp VLA -> planner->basket -> place VLA

Each panel shows the representative frame at the END of its phase, with the
gripper trajectory drawn up to that moment: past phases thin/faded, the
current phase thick. Planner phases are slate gray, VLA phases are accent
orange — matching the page's chart palette.

The gripper track comes from SAM3 "robot arm" masks sampled every 20 frames
(bottom-of-mask point), stored in arm_track.npy as rows [frame_idx, x, y].

Usage:
  python3 make_phase_strip.py input.mov --track arm_track.npy -o phase_strip.png
"""
import argparse
import cv2
import numpy as np

SLATE = (114, 100, 91)     # BGR of #5B6472
ORANGE = (61, 106, 255)    # BGR of #FF6A3D
WHITE = (255, 255, 255)

# (label, end_anchor_frame, color, is_vla) — post-stride indices
PHASES = [
    ("home",                 0,   None,   False),
    ("transport → object",   100, SLATE,  False),
    ("grasp VLA",            270, ORANGE, True),
    ("transport → basket",   350, SLATE,  False),
    ("place VLA",            421, ORANGE, True),
]

CROP = (380, 40, 1520, 1080)  # x0, y0, x1, y1 — action region of the 1920x1080 frame


def read_frames(path, stride=2):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def resample(track, n=400):
    """Densify the anchor polyline with linear interpolation + smoothing."""
    f = track[:, 0]
    # de-jitter anchors first (SAM3 bottom-point wobbles near the end)
    tx = np.convolve(np.pad(track[:, 1], 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    ty = np.convolve(np.pad(track[:, 2], 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    t = np.linspace(f[0], f[-1], n)
    x = np.interp(t, f, tx)
    y = np.interp(t, f, ty)
    k = 15
    ker = np.ones(k) / k
    x = np.convolve(np.pad(x, k // 2, mode="edge"), ker, mode="valid")
    y = np.convolve(np.pad(y, k // 2, mode="edge"), ker, mode="valid")
    return t, x, y


def _poly(img, pts, color, thick):
    """Line with a white casing so it stays legible on any background."""
    cv2.polylines(img, [pts], False, WHITE, thick + 6, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


def _arrow(img, p0, p1, color):
    """Direction arrowhead at p1 pointing along p0->p1."""
    v = np.array(p1, float) - np.array(p0, float)
    n = np.linalg.norm(v)
    if n < 1: return
    v /= n
    tip = np.array(p1, float)
    left = tip - 26 * v + 14 * np.array([-v[1], v[0]])
    right = tip - 26 * v - 14 * np.array([-v[1], v[0]])
    tri = np.array([tip, left, right], np.int32)
    cv2.fillPoly(img, [tri], WHITE, cv2.LINE_AA)
    tri_in = np.array([tip - 3 * v, tip - 22 * v + 10 * np.array([-v[1], v[0]]),
                       tip - 22 * v - 10 * np.array([-v[1], v[0]])], np.int32)
    cv2.fillPoly(img, [tri_in], color, cv2.LINE_AA)


def _lighten(color, f=0.5):
    return tuple(int(c + (255 - c) * f) for c in color)


def draw_trace(img, t, x, y, t_end):
    """Draw the phase-colored trace up to time t_end onto img (in place).

    Past phases: thin, whitened tint — context, not the subject.
    Current phase: thick cased line + arrowhead + endpoint dot.
    """
    prev_end = PHASES[0][1]
    current = None
    for label, end, color, _ in PHASES[1:]:
        if color is None:
            prev_end = end
            continue
        seg = (t >= prev_end) & (t <= min(end, t_end))
        if seg.sum() >= 2:
            pts = np.stack([x[seg], y[seg]], axis=1).astype(np.int32)
            if end >= t_end:
                current = (pts, color)
            else:
                cv2.polylines(img, [pts], False, _lighten(color), 5, cv2.LINE_AA)
        if end >= t_end:
            break
        prev_end = end
    if current is not None:
        pts, color = current
        _poly(img, pts, color, 10)
        if len(pts) >= 12:
            _arrow(img, pts[-12], pts[-1], color)
        cx, cy = np.median(pts[-5:], axis=0).astype(int)
        cv2.circle(img, (cx, cy), 14, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 10, color, -1, cv2.LINE_AA)


def label_panel(img, text, color):
    """Pill label in the top-left corner."""
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    (tw, tth), _ = cv2.getTextSize(text, font, scale, th)
    pad = 18
    x0, y0 = 24, 24
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + tth + 2 * pad), WHITE, -1)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + tth + 2 * pad), color, 3)
    cv2.putText(img, text, (x0 + pad, y0 + pad + tth - 4), font, scale, (30, 33, 40), th, cv2.LINE_AA)


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--track", required=True, help="arm_track.npy [frame,x,y] rows")
    ap.add_argument("-o", "--output", help="combined strip PNG")
    ap.add_argument("--panels-dir", help="write individual panels (no baked labels) here for HTML layout")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    frames = read_frames(args.input, args.stride)
    track = np.load(args.track)
    t, x, y = resample(track)

    x0, y0, x1, y1 = CROP
    panels = []
    for i, (label, end, color, _) in enumerate(PHASES):
        idx = min(end, len(frames) - 1)
        panel = frames[idx].copy()
        if end > 0:
            draw_trace(panel, t, x, y, end)
        panel = panel[y0:y1, x0:x1]
        if args.panels_dir:
            os.makedirs(args.panels_dir, exist_ok=True)
            # web-sized, no baked text (label lives in HTML for crisp type)
            web = cv2.resize(panel, (640, int(640 * panel.shape[0] / panel.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(args.panels_dir, f"phase_{i}.jpg"), web, [cv2.IMWRITE_JPEG_QUALITY, 88])
        labeled = panel.copy()
        label_panel(labeled, label, color or (150, 150, 150))
        panels.append(labeled)

    if args.panels_dir:
        print("wrote", len(PHASES), "panels to", args.panels_dir)

    if args.output:
        gap = 14
        h = panels[0].shape[0]
        strip = np.full((h, sum(p.shape[1] for p in panels) + gap * (len(panels) - 1), 3), 255, np.uint8)
        cx = 0
        for p in panels:
            strip[:, cx:cx + p.shape[1]] = p
            cx += p.shape[1] + gap
        cv2.imwrite(args.output, strip)
        print("wrote", args.output, strip.shape)


if __name__ == "__main__":
    main()
