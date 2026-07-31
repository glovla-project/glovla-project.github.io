#!/usr/bin/env python3
"""Render example LIBERO-Challenge scenes as first-frame images (Fig. 2 style).

Runs headless (MUJOCO_GL=egl or osmesa) in the `libero` conda env, reusing
HybridVLA's robustness machinery — no policy or checkpoint needed.

  MUJOCO_GL=egl /home/vinit-admin/miniconda3/envs/libero/bin/python \
      tools/render_challenge_scenes.py --out /tmp/scenes

Scenes rendered (matching the paper's Fig. 2 layout):
  (a)-(e) easy:   clutter, distraction, illumination, obstruction, visual shift
  (f),(g) medium: 2-3 combined perturbations
  (h)-(j) hard:   4-5 combined perturbations
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

HYBRIDVLA_SCRIPTS = "/home/vinit-admin/truongnt/HybridVLA/scripts"
sys.path.insert(0, HYBRIDVLA_SCRIPTS)

import numpy as np

SCENES = [
    ("a_clutter",      "clutter_scene1"),
    ("b_distraction",  "distraction_scene1"),
    ("c_illumination", "illumination_scene1"),
    ("d_obstruction",  "obstruction_scene3"),
    ("e_visualshift",  "visualshift_scene1"),
    ("f_medium1",      "medium_scene1"),
    ("g_medium4",      "medium_scene4"),
    ("h_difficult2",   "difficult_scene2"),
    ("i_difficult1",   "difficult_scene1"),
    ("j_difficult6",   "difficult_scene6"),
]

TASK = "pick_up_the_chocolate_pudding_and_place_it_in_the_basket"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--wait-steps", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import cv2
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_robustness import prepare_libero_robustness_env

    def get_dummy_action():
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    def find_task(task_suite_name, wanted):
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.name == wanted or task.language == wanted:
                return task_suite, task_id, task
        raise ValueError("task not found: " + wanted)

    task_suite, task_id, task = find_task("libero_object", TASK)
    initial_states = task_suite.get_task_init_states(task_id)

    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    for label, mode in SCENES:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=args.resolution,
            camera_widths=args.resolution,
        )
        env.seed(0)
        env.reset()
        obs, _ = prepare_libero_robustness_env(
            env,
            initial_states[0],
            mode=mode,
            object_obs_key="chocolate_pudding_1_pos",
            object_body_name="chocolate_pudding_1_main",
            seed=0,
            severity=1.0,
        )
        for _ in range(args.wait_steps):
            obs, _, _, _ = env.step(get_dummy_action())
        # agentview flipped exactly like the eval pipeline; visual-shift image
        # post-processing (if any) is baked in by process_libero_images inside
        # libero_robustness at prepare time, so the raw agentview is what the
        # policy sees for physical scenes
        image = obs["agentview_image"][::-1, ::-1].astype(np.uint8)
        try:
            from libero_robustness import process_libero_images
            image, _ = process_libero_images(
                image, obs["robot0_eye_in_hand_image"][::-1, ::-1].astype(np.uint8),
                mode=mode, severity=1.0)
        except ImportError:
            pass
        out_path = os.path.join(args.out, f"{label}.png")
        cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("wrote", out_path, flush=True)
        env.close()


if __name__ == "__main__":
    main()
