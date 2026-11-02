#!/usr/bin/env python3
"""Evaluate baseline, BC, and BC+RL on the held-out character only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from bc_train import BehaviorCloningPolicy
from brush_env import BrushCalligraphyEnv, validate_observation_checkpoint
from calligraphy_config import (
    DEFAULT_BASELINE_BRUSH_WIDTH_PX,
    DEFAULT_DATA_DIR,
    HELD_OUT_CHARACTER,
    PROJECT_ROOT,
    TRAIN_CHARACTERS,
)
from rl_train import PPOPolicy


def load_bc_policy(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_observation_checkpoint(checkpoint)
    if set(checkpoint["characters"]) != set(TRAIN_CHARACTERS):
        raise ValueError("BC checkpoint must contain exactly the three training characters")
    policy = BehaviorCloningPolicy(
        observation_size=checkpoint["observation_size"],
        hidden_size=checkpoint["hidden_size"],
        action_size=checkpoint["action_size"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return checkpoint, policy


def load_rl_policy(checkpoint_path, bc_checkpoint):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_observation_checkpoint(checkpoint)
    if set(checkpoint["characters"]) != set(TRAIN_CHARACTERS):
        raise ValueError("BC+RL checkpoint must contain exactly the three training characters")
    if checkpoint.get("algorithm") != "PPO initialized from BC":
        raise ValueError("BC+RL checkpoint metadata is missing or unexpected")
    policy = PPOPolicy(bc_checkpoint)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return checkpoint, policy


def corrected_reference(env, index):
    """Return the CSV reference shifted toward the target stroke center."""
    reference = env.trajectory[index, :2].copy()
    if env.contact[index]:
        tangent = env._tangent(index)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        offset = np.clip(
            env.target_cross_sections[index, 0], -env.max_xy_offset_px, env.max_xy_offset_px
        )
        reference += normal * offset
    return reference


def rollout(env, policy_name, policy=None, character=HELD_OUT_CHARACTER):
    observation, _ = env.reset(options={"character": character})
    baseline_pressure = env.pressure_for_brush_width(
        DEFAULT_BASELINE_BRUSH_WIDTH_PX
    )
    trace_rows = []
    terminated = False
    while not terminated:
        index = env.step_index
        reference = corrected_reference(env, index)
        if policy_name == "baseline":
            offset = (reference - env.trajectory[index, :2]) / env.max_xy_offset_px
            action = np.array([*offset, baseline_pressure], dtype=np.float32)
        elif policy_name == "bc":
            with torch.no_grad():
                action = policy(
                    torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy()
        else:
            with torch.no_grad():
                action = policy.deterministic_action(
                    torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy()
            action[2] = (action[2] + 1.0) / 2.0

        observation, _, terminated, _, info = env.step(action)
        trace_rows.append({
            "step": env.step_index - 1,
            "stroke_id": info["stroke_id"],
            "contact": int(info["contact"]),
            "reference_x_px": float(reference[0]),
            "reference_y_px": float(reference[1]),
            "actual_x_px": float(env.position[0]),
            "actual_y_px": float(env.position[1]),
            "pressure": float(info["pressure"]),
            "brush_width_px": int(info["brush_width_px"]),
        })

    return Image.fromarray(env.render()).convert("L"), {
        key: float(info[key])
        for key in ("dice", "overflow_rate", "unpainted_rate")
    }, trace_rows


def save_comparison(images, output_path):
    labels = ("Target", "Baseline", "BC", "BC + RL")
    panel_size = 256
    margin = 18
    title_height = 28
    canvas = Image.new(
        "RGB",
        (len(labels) * panel_size + (len(labels) + 1) * margin, panel_size + title_height + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = margin + index * (panel_size + margin)
        draw.text((x, margin), label, fill="#20252b")
        canvas.paste(image.convert("RGB").resize((panel_size, panel_size)), (x, margin + title_height))
        draw.rectangle(
            (x, margin + title_height, x + panel_size - 1, margin + title_height + panel_size - 1),
            outline="#9aa2a9",
        )
    canvas.save(output_path)


def split_contact_strokes(rows):
    strokes = []
    current_id = None
    current_rows = []
    for row in rows:
        stroke_id = int(row["stroke_id"])
        if not row["contact"] or not stroke_id:
            if current_rows:
                strokes.append(current_rows)
                current_rows = []
            current_id = None
            continue
        if current_id is not None and stroke_id != current_id:
            strokes.append(current_rows)
            current_rows = []
        current_id = stroke_id
        current_rows.append(row)
    if current_rows:
        strokes.append(current_rows)
    return strokes


def draw_dashed_path(draw, points, color, width, dash_length=8, gap_length=5):
    for start, end in zip(points, points[1:]):
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length = float(np.hypot(delta_x, delta_y))
        if length == 0:
            continue
        offset = 0.0
        while offset < length:
            dash_end = min(offset + dash_length, length)
            first = (
                start[0] + delta_x * offset / length,
                start[1] + delta_y * offset / length,
            )
            second = (
                start[0] + delta_x * dash_end / length,
                start[1] + delta_y * dash_end / length,
            )
            draw.line((first, second), fill=color, width=width)
            offset += dash_length + gap_length


def summary_font(size):
    """Load a scalable font with Japanese glyphs for summary labels."""
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise OSError("A Japanese font is required to render summary labels")


def render_trajectory_overlay(target, rows, label):
    scale = 2
    header_height = 74
    actual_color = "#a04d9b"
    target_rgb = target.convert("RGB").resize(
        (target.width * scale, target.height * scale), Image.NEAREST
    )
    faded_target = Image.blend(
        target_rgb,
        Image.new("RGB", target_rgb.size, "white"),
        0.72,
    )
    canvas = Image.new("RGB", (faded_target.width, faded_target.height + header_height), "white")
    canvas.paste(faded_target, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 5), f"{label}: reference vs executed", fill="#20252b", font=summary_font(20))
    draw.line((14, 53, 48, 53), fill="#2471a3", width=2)
    draw.text((56, 40), "corrected reference", fill="#20252b", font=summary_font(16))
    draw.line((285, 53, 319, 53), fill=actual_color, width=3)
    draw.text((327, 40), "executed", fill="#20252b", font=summary_font(16))

    def scaled_points(stroke, x_key, y_key):
        return [
            (float(row[x_key]) * scale, float(row[y_key]) * scale + header_height)
            for row in stroke
        ]

    strokes = split_contact_strokes(rows)
    for stroke in strokes:
        executed_points = scaled_points(stroke, "actual_x_px", "actual_y_px")
        if len(executed_points) > 1:
            draw.line(executed_points, fill=actual_color, width=3, joint="curve")
        elif executed_points:
            x, y = executed_points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=actual_color)

    for stroke in strokes:
        reference_points = scaled_points(stroke, "reference_x_px", "reference_y_px")
        draw_dashed_path(draw, reference_points, "#2471a3", width=2)

    return canvas


def save_trajectory_overlays(
    target, trajectories, output_dir, character=HELD_OUT_CHARACTER
):
    labels = {
        "baseline": "Baseline",
        "bc": "BC",
        "bc_rl": "BC + RL",
    }
    panels = []
    for name, rows in trajectories.items():
        label = labels[name]
        panel = render_trajectory_overlay(target, rows, label)
        panel.save(output_dir / f"{name}_{character}_trajectory_overlay.png")
        panels.append(panel)

    margin = 16
    comparison = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + margin * (len(panels) + 1), panels[0].height + 2 * margin),
        "white",
    )
    x = margin
    for panel in panels:
        comparison.paste(panel, (x, margin))
        x += panel.width + margin
    comparison_name = (
        "trajectory_overlay_comparison.png"
        if character == HELD_OUT_CHARACTER
        else f"trajectory_overlay_comparison_{character}.png"
    )
    comparison.save(output_dir / comparison_name)


def evaluate(args):
    if HELD_OUT_CHARACTER in TRAIN_CHARACTERS:
        raise ValueError("Held-out character must remain separate from training characters")
    bc_checkpoint, bc_policy = load_bc_policy(args.bc_checkpoint)
    _rl_checkpoint, rl_policy = load_rl_policy(args.rl_checkpoint, bc_checkpoint)
    env = BrushCalligraphyEnv(
        artifact_dir=args.data_dir,
        characters=(HELD_OUT_CHARACTER,),
    )
    if env.character_names != (HELD_OUT_CHARACTER,):
        raise ValueError("Evaluation environment must load only the held-out character")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = env.datasets[HELD_OUT_CHARACTER]["target"].convert("L")
    results = {}
    trajectories = {}
    images = [target]
    for name, policy in (
        ("baseline", None),
        ("bc", bc_policy),
        ("bc_rl", rl_policy),
    ):
        image, metrics, rows = rollout(env, name, policy)
        image.save(args.output_dir / f"{name}_{HELD_OUT_CHARACTER}.png")
        results[name] = metrics
        trajectories[name] = rows
        images.append(image)
        with (args.output_dir / f"{name}_{HELD_OUT_CHARACTER}_trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    save_comparison(images, args.output_dir / "comparison.png")
    save_trajectory_overlays(target, trajectories, args.output_dir)
    metrics = {
        "evaluation_character": HELD_OUT_CHARACTER,
        "training_characters": list(TRAIN_CHARACTERS),
        "held_out_data_used_for_training_or_tuning": False,
        "bc_checkpoint": str(args.bc_checkpoint),
        "bc_rl_checkpoint": str(args.rl_checkpoint),
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--bc-checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts" / "day5_day4_self_corner_fixed" / "bc_policy.pt",
    )
    parser.add_argument(
        "--rl-checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts" / "day7_rl" / "bc_rl_policy.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "day8_heldout",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())