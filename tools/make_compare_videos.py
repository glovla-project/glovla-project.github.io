#!/usr/bin/env python3
"""Stack FullVLA (top) vs GloVLA (bottom) LIBERO rollouts into labeled compare clips.

Each source video is 512x256 = [third-person | wrist] side by side. We crop the
left (third-person) 256x256 half, stack the two methods vertically, and burn in a
method label + SUCCESS/FAIL badge (read from the filename's success_<0|1> token).

Output: static/videos/sim/<scene>.mp4  (one per scene)

Run with the project's ffmpeg; needs the drawtext filter (standard build).
"""
import argparse
import glob
import os
import re
import subprocess

BASE = ("/home/vinit-admin/truongnt/HybridVLA-Manifold/outputs/LIBERO/"
        "robustness_video_compare/chocolate_pudding/20260731_160714")

# scene key -> nice label + difficulty tier
SCENES = [
    ("clutter_scene1",     "Clutter",          "easy"),
    ("distraction_scene1", "Distraction",      "easy"),
    ("obstruction_scene1", "Obstruction",      "easy"),
    ("visualshift_scene1", "Visual shift",     "easy"),
    ("illumination_scene1","Illumination",     "easy"),
    ("medium_scene1",      "Medium comp. 1",   "medium"),
    ("medium_scene2",      "Medium comp. 2",   "medium"),
    ("medium_scene3",      "Medium comp. 3",   "medium"),
    ("difficult_scene1",   "Hard comp. 1",     "hard"),
    ("difficult_scene2",   "Hard comp. 2",     "hard"),
]


def find_video(scene, method):
    hits = glob.glob(os.path.join(BASE, scene, method, "videos", "*.mp4"))
    return hits[0] if hits else None


def success_of(path):
    m = re.search(r"success_(\d)", os.path.basename(path))
    return m and m.group(1) == "1"


def build(scene, label, out_path, crop_left, scale_w):
    full = find_video(scene, "fullvla")
    hyb = find_video(scene, "hybridvla")
    if not full or not hyb:
        print("skip", scene, "(missing pair)")
        return
    full_ok = success_of(full)
    hyb_ok = success_of(hyb)

    # crop: left half (third-person) if requested, else keep full frame
    crop = "crop=256:256:0:0," if crop_left else ""
    w = scale_w
    # per-input filter: crop -> scale -> pad a top strip for the label bar
    def lane(ok, name):
        badge = "SUCCESS" if ok else "FAILURE"
        bcol = "0x1Fab7a" if ok else "0xd03b3b"
        return (
            f"{crop}scale={w}:-2,"
            f"drawbox=x=0:y=0:w=iw:h=40:color=0x14171c@0.85:t=fill,"
            f"drawtext=text='{name}':x=16:y=10:fontsize=22:fontcolor=white,"
            f"drawtext=text='{badge}':x=w-tw-16:y=10:fontsize=22:fontcolor={bcol}"
        )

    filt = (
        f"[0:v]{lane(full_ok, 'FullVLA')}[top];"
        f"[1:v]{lane(hyb_ok, 'GloVLA (Ours)')}[bot];"
        f"[top][bot]vstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-stream_loop", "-1", "-t", "9.5", "-i", full,
        "-stream_loop", "-1", "-t", "9.5", "-i", hyb,
        "-filter_complex", filt, "-map", "[v]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "24",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    print("wrote", out_path, "| full", "OK" if full_ok else "FAIL",
          "| glo", "OK" if hyb_ok else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="static/videos/sim")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--keep-wrist", action="store_true", help="keep both views (no left crop)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for scene, label, tier in SCENES:
        out = os.path.join(args.out_dir, f"{scene}.mp4")
        build(scene, label, out, crop_left=not args.keep_wrist, scale_w=args.width)


if __name__ == "__main__":
    main()
