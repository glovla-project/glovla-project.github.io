#!/usr/bin/env python3
"""Stroboscopic trajectory figure: solid robot keyframes composited over a clean plate.

Pipeline (static tripod camera assumed):
  1. Clean background plate = first frame with its home-pose robot patched
     from the temporal median (the median alone gets contaminated wherever the
     robot dwells, so neither source is usable by itself).
  2. Per-keyframe robot mask — two modes:
       * default: shadow-suppressed background difference (max-channel +
         Lab-chroma test) with fragment-gluing. Fails where the silver arm
         overlaps the white wall (diff ~ 0).
       * --masks-dir: pre-computed SAM3 masks named sam3_mask_<origidx>.png
         (text prompts "robot arm" + "blue cup"). NOTE: for videos carrying
         rotation metadata (e.g. iPhone .mov rotation=-180) the SAM3 masks
         come back 180-degree rotated relative to OpenCV frames — they are
         auto-rotated here. Holes filled, edges closed.
  3. Keyframes pasted fully opaque in chronological order (later poses on
     top); 2px seam blur along mask boundaries.

Usage (bg-diff):   python3 make_strobe.py input.mov -o strobe.png --frames 230,537,650,790:16
Usage (SAM3):      python3 make_strobe.py input.mov -o strobe.png --frames 230,600,790 \
                       --masks-dir tools/sam3_masks
(--frames indices are ORIGINAL video frame numbers; masks in --masks-dir are
named by post-stride index = orig // stride.)

The official figure was produced with SAM3 masks (keys 115,300,395 post-stride,
i.e. --frames 230,600,790 --masks-dir tools/sam3_masks). To regenerate masks,
run SAM3 in the `sam3` conda env with checkpoint
/home/vinit-admin/truongnt/checkpoints/sam3/sam3.pt.
"""
import argparse
import cv2
import numpy as np


def read_frames(path, end_sec=None, stride=1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    end_frame = int(end_sec * fps) if end_sec else None
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (end_frame is not None and idx >= end_frame):
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps, stride


def median_background(frames, max_samples=60):
    step = max(1, len(frames) // max_samples)
    stack = np.stack(frames[::step]).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def clean_plate(frames):
    """First frame with its (home-pose) robot patched from the median."""
    f0 = frames[0].astype(np.float32)
    med = median_background(frames).astype(np.float32)
    d0 = np.abs(f0 - med).max(axis=2)
    m0 = (cv2.GaussianBlur(d0, (9, 9), 0) > 24).astype(np.uint8) * 255
    m0 = cv2.dilate(m0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m0)
    keep = np.zeros_like(m0)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30000:
            keep[labels == i] = 255
    mm = (cv2.GaussianBlur(keep, (21, 21), 0).astype(np.float32) / 255.0)[..., None]
    return (f0 * (1 - mm) + med * mm).astype(np.uint8)


def solid_mask(frame, background, thresh=26, min_area=3000):
    f = frame.astype(np.int16)
    b = background.astype(np.int16)
    max_ch = np.abs(f - b).max(axis=2)
    lab_f = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.int16)
    lab_b = cv2.cvtColor(background, cv2.COLOR_BGR2Lab).astype(np.int16)
    chroma = np.abs(lab_f[..., 1] - lab_b[..., 1]) + np.abs(lab_f[..., 2] - lab_b[..., 2])
    raw = ((max_ch > thresh * 1.6) | ((max_ch > thresh) & (chroma > 8))).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    # glue arm fragments before filtering so occlusion-split pieces survive
    big = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(big)
    keepids = {i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 60000}
    glue = np.isin(labels, list(keepids)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(raw, glue)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k, iterations=4)
    holes = keep.copy()
    h, w = holes.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(holes, ff, (0, 0), 255)
    return keep | cv2.bitwise_not(holes)


def load_sam3_mask(masks_dir, idx, frame=None, plate=None):
    """SAM3 mask refined with background-diff detail.

    SAM3 gets the big regions right (arm on white wall) but chews thin dark
    structures (gripper fingers). Background diff is the opposite. So: take
    the SAM3 mask, dilate a 35px envelope around it, and add any pixel inside
    the envelope whose diff against the clean plate exceeds a low threshold —
    detail recovery without wall leakage.
    """
    import os
    path = os.path.join(masks_dir, f"sam3_mask_{idx}.png")
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    # SAM3 masks are 180-deg rotated vs OpenCV frames for rotation-tagged videos
    m = cv2.rotate(m, cv2.ROTATE_180)
    sam = (m > 127).astype(np.uint8) * 255
    if frame is not None and plate is not None:
        env = cv2.dilate(sam, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
        d = np.abs(frame.astype(np.int16) - plate.astype(np.int16)).max(axis=2)
        detail = ((d > 20) & (env > 0)).astype(np.uint8) * 255
        sam = sam | detail
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sam = cv2.morphologyEx(sam, cv2.MORPH_CLOSE, k, iterations=3)
    holes = (sam > 127).astype(np.uint8) * 255
    h, w = holes.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    filled = holes.copy()
    cv2.floodFill(filled, ff, (0, 0), 255)
    sam = holes | cv2.bitwise_not(filled)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sam)
    keep = np.zeros_like(sam)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] >= 800:
            keep[labels == c] = 255
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--frames", required=True,
                    help="comma-separated original frame indices; append :thresh to override per keyframe (e.g. 790:16)")
    ap.add_argument("--end-sec", type=float, default=None)
    ap.add_argument("--thresh", type=int, default=26)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--masks-dir", default=None,
                    help="use pre-computed SAM3 masks (sam3_mask_<postStrideIdx>.png) instead of bg-diff")
    args = ap.parse_args()

    frames, fps, stride = read_frames(args.input, args.end_sec, args.stride)
    print(f"read {len(frames)} frames (stride {stride}, fps {fps:.1f})")
    bg = clean_plate(frames)
    bgf = bg.astype(np.float32)

    specs = []
    for tok in args.frames.split(","):
        if ":" in tok:
            i, th = tok.split(":")
            specs.append((min(int(i) // stride, len(frames) - 1), int(th)))
        else:
            specs.append((min(int(tok) // stride, len(frames) - 1), args.thresh))
    print("keyframes (post-stride idx, thresh):", specs)

    canvas = bgf.copy()
    if args.masks_dir:
        masks = [load_sam3_mask(args.masks_dir, i, frames[i], bg) for i, _ in specs]
    else:
        masks = [solid_mask(frames[i], bg, thresh=th) for i, th in specs]
    for (i, _), m in zip(specs, masks):
        sel = m.astype(bool)
        canvas[sel] = frames[i].astype(np.float32)[sel]

    edge = np.zeros(bg.shape[:2], np.uint8)
    for m in masks:
        edge |= cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    blur = cv2.GaussianBlur(canvas, (5, 5), 0)
    eb = (cv2.GaussianBlur(edge, (7, 7), 0).astype(np.float32) / 255.0)[..., None]
    canvas = canvas * (1 - eb) + blur * eb

    cv2.imwrite(args.output, canvas.astype(np.uint8))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
