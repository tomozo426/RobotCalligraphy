#!/usr/bin/env python3
"""Render baseline, BC, and BC+RL trajectory overlays for a training character."""

import argparse
from pathlib import Path

from brush_env import BrushCalligraphyEnv
from calligraphy_config import DEFAULT_DATA_DIR, PROJECT_ROOT, TRAIN_CHARACTERS
from evaluate_heldout import (
    load_bc_policy,
    load_rl_policy,
    rollout,
    save_trajectory_overlays,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", choices=TRAIN_CHARACTERS, default="ichi")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "day5_day4_self_corner_fixed" / "bc_policy.pt",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "day7_rl" / "bc_rl_policy.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "day8_ichi_visualization",
    )
    return parser.parse_args()


def main(args):
    bc_checkpoint, bc_policy = load_bc_policy(args.bc_checkpoint)
    _, rl_policy = load_rl_policy(args.rl_checkpoint, bc_checkpoint)
    env = BrushCalligraphyEnv(artifact_dir=args.data_dir, characters=(args.character,))
    target = env.datasets[args.character]["target"].convert("L")
    trajectories = {}
    for name, policy in (
        ("baseline", None),
        ("bc", bc_policy),
        ("bc_rl", rl_policy),
    ):
        _, _, rows = rollout(env, name, policy, character=args.character)
        trajectories[name] = rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_overlays(
        target, trajectories, args.output_dir, character=args.character
    )
    print(f"Saved trajectory overlays to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())