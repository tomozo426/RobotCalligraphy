#!/usr/bin/env python3
"""Render a visual diagnosis of the BC policy's final brush stroke."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from bc_train import BehaviorCloningPolicy, teacher_action
from brush_env import BrushCalligraphyEnv, validate_observation_checkpoint
from calligraphy_config import (
    ACTION_SIZE,
    CONTROL_HZ,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MULTICHAR_ARTIFACT_DIR,
    MAX_BRUSH_WIDTH_PX,
    MAX_PRESSURE_RATE_PER_SECOND,
    MAX_XY_OFFSET_PX,
    MIN_BRUSH_WIDTH_PX,
    OBSERVATION_SIZE,
    POLICY_HIDDEN_SIZE,
    PROJECT_ROOT,
    TRAIN_CHARACTERS,
)


def load_font(size):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rollout(artifact_dir, policy=None, env_kwargs=None):
    env = BrushCalligraphyEnv(artifact_dir=artifact_dir, **(env_kwargs or {}))
    observation, _ = env.reset()
    actions = []
    sample_positions = []
    actual_pressures = []
    terminated = False
    while not terminated:
        step_index = env.step_index
        if policy is None:
            action = teacher_action(env, step_index)
        else:
            with torch.no_grad():
                observation_size = policy.network[0].in_features
                state = torch.as_tensor(
                    observation[:observation_size], dtype=torch.float32
                ).unsqueeze(0)
                action = policy(state).squeeze(0).numpy()
        actions.append(action.copy())
        observation, _, terminated, _, info = env.step(action)
        sample_positions.append(env.position.copy())
        actual_pressures.append(info["pressure"])
    return (
        Image.fromarray(env.render()).convert("L"),
        np.asarray(actions),
        np.asarray(sample_positions),
        np.asarray(actual_pressures),
        env,
    )


def add_full_view(canvas, draw, title, image, x, y):
    draw.text((x, y), title, fill="#20252b", font=load_font(22))
    resized = image.resize((320, 320), Image.NEAREST).convert("RGB")
    canvas.paste(resized, (x + 50, y + 36))
    draw.rectangle((x + 50, y + 36, x + 370, y + 356), outline="#aeb5bc", width=1)


def endpoint_overlay(target_mask, prediction, crop, scale=10):
    left, top, right, bottom = crop
    target = target_mask[top:bottom, left:right]
    predicted = np.asarray(prediction)[top:bottom, left:right] < 128
    pixels = np.empty((*target.shape, 3), dtype=np.uint8)
    pixels[target & predicted] = (24, 31, 38)
    pixels[target & ~predicted] = (169, 181, 192)
    pixels[~target & predicted] = (235, 82, 48)
    pixels[~target & ~predicted] = (255, 255, 255)
    image = Image.fromarray(pixels).resize(
        ((right - left) * scale, (bottom - top) * scale), Image.NEAREST
    )
    return image


def add_zoom_panel(canvas, draw, title, prediction, target_mask, crop, sample_positions, x, y):
    draw.text((x, y), title, fill="#20252b", font=load_font(20))
    overlay = endpoint_overlay(target_mask, prediction, crop)
    canvas.paste(overlay, (x, y + 32))
    draw.rectangle(
        (x, y + 32, x + overlay.width, y + 32 + overlay.height),
        outline="#aeb5bc",
        width=1,
    )
    left, top, _, _ = crop
    for index in range(max(0, len(sample_positions) - 10), len(sample_positions)):
        point_x, point_y = sample_positions[index]
        center_x = x + round((float(point_x) - left) * 10)
        center_y = y + 32 + round((float(point_y) - top) * 10)
        draw.ellipse(
            (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
            fill="#00a6c7",
            outline="white",
            width=2,
        )
        label_y = center_y - 20 if index % 2 == 0 else center_y + 7
        draw.rectangle((center_x - 9, label_y, center_x + 9, label_y + 12), fill="white")
        draw.text((center_x - 7, label_y), str(index), fill="#006d86", font=load_font(10))


def add_pressure_plot(
    canvas,
    draw,
    title,
    trajectory,
    target_mask,
    teacher_actions,
    teacher_actual_pressures,
    policy_actions,
    policy_actual_pressures,
    env,
    x,
    y,
):
    draw.text((x, y), title, fill="#20252b", font=load_font(20))
    plot = (x + 48, y + 62, x + 398, y + 302)
    draw.rectangle(plot, outline="#7d858d", width=1)
    for tick in range(5):
        value = tick / 4
        py = plot[3] - value * (plot[3] - plot[1])
        draw.line((plot[0], py, plot[2], py), fill="#e3e6e8", width=1)
        draw.text((x + 8, py - 7), f"{value:.2f}", fill="#50575e", font=load_font(12))

    indices = np.arange(len(trajectory))
    target_pressures = []
    for index in indices:
        column = int(round(float(trajectory[index, 0])))
        width = int(target_mask[:, column].sum())
        target_pressures.append(env.pressure_for_brush_width(width))

    series = (
        (target_pressures, "#687681", "target width"),
        (teacher_actions[indices, 2], "#e28a25", "teacher command"),
        (teacher_actual_pressures[indices], "#168f88", "teacher actual"),
        (policy_actions[indices, 2], "#d44343", "BC command"),
        (policy_actual_pressures[indices], "#2769a6", "BC actual"),
    )
    x_positions = [
        plot[0] + offset * (plot[2] - plot[0]) / max(len(indices) - 1, 1)
        for offset in range(len(indices))
    ]
    for values, color, _ in series:
        points = [
            (round(px), round(plot[3] - float(value) * (plot[3] - plot[1])))
            for px, value in zip(x_positions, values)
        ]
        draw.line(points, fill=color, width=3)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)

    for offset, index in enumerate(indices):
        if offset % 5 == 0 or offset == len(indices) - 1:
            draw.text(
                (x_positions[offset] - 6, plot[3] + 8),
                str(index),
                fill="#50575e",
                font=load_font(10),
            )
    draw.text(
        (x + 108, y + 350),
        f"Brush pressure by step (all {len(trajectory)} steps)",
        fill="#50575e",
        font=load_font(13),
    )

    legend_y = y + 386
    for index, (values, color, label) in enumerate(series):
        legend_x = x + 6 + (index % 3) * 132
        legend_row = index // 3
        row_y = legend_y + legend_row * 20
        draw.line((legend_x, row_y + 7, legend_x + 18, row_y + 7), fill=color, width=3)
        draw.text((legend_x + 23, row_y), label, fill="#343a40", font=load_font(11))


def rollout_full_character(env, character, policy=None):
    observation, _ = env.reset(options={"character": character})
    actions = []
    actual_pressures = []
    target_width_ratios = []
    terminated = False
    while not terminated:
        step_index = env.step_index
        target_width_ratios.append(float(observation[12]))
        if policy is None:
            action = teacher_action(env, step_index)
        else:
            with torch.no_grad():
                observation_size = policy.network[0].in_features
                state = torch.as_tensor(
                    observation[:observation_size], dtype=torch.float32
                ).unsqueeze(0)
                action = policy(state).squeeze(0).numpy()
        actions.append(action.copy())
        observation, _, terminated, _, info = env.step(action)
        actual_pressures.append(float(info["pressure"]))

    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "actual_pressures": np.asarray(actual_pressures, dtype=np.float32),
        "target_width_ratios": np.asarray(target_width_ratios, dtype=np.float32),
        "trajectory": env.trajectory.copy(),
        "contact": env.contact.copy(),
        "stroke_ids": env.stroke_ids.copy(),
    }


def save_full_pressure_csv(teacher, policy, control_hz, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "step",
                "time_s",
                "stroke_id",
                "contact",
                "target_width_ratio",
                "teacher_command",
                "teacher_actual",
                "bc_command",
                "bc_actual",
            )
        )
        for index in range(len(teacher["trajectory"])):
            writer.writerow(
                (
                    index,
                    index / control_hz,
                    int(teacher["stroke_ids"][index]),
                    int(teacher["contact"][index]),
                    float(teacher["target_width_ratios"][index]),
                    float(teacher["actions"][index, 2]),
                    float(teacher["actual_pressures"][index]),
                    float(policy["actions"][index, 2]),
                    float(policy["actual_pressures"][index]),
                )
            )


def create_full_pressure_plot(character, teacher, policy, control_hz, pressure_rate, output_path):
    width, height = 1280, 640
    plot = (100, 78, 1230, 470)
    image = Image.new("RGB", (width, height), "#f7f8f9")
    draw = ImageDraw.Draw(image)
    step_count = len(teacher["trajectory"])

    def x_position(index):
        return round(plot[0] + index * (plot[2] - plot[0]) / max(step_count - 1, 1))

    def y_position(value):
        return round(plot[3] - float(np.clip(value, 0.0, 1.0)) * (plot[3] - plot[1]))

    draw.text((42, 22), f"Full-trajectory pressure: {character}", fill="#171b1f", font=load_font(28))
    draw.text(
        (42, 54),
        f"{step_count} control steps | {step_count / control_hz:.1f} s | {len(set(teacher['stroke_ids']) - {0})} strokes",
        fill="#59636c",
        font=load_font(15),
    )

    contact = teacher["contact"]
    lift_indices = np.flatnonzero(~contact)
    if len(lift_indices):
        lift_runs = np.split(lift_indices, np.flatnonzero(np.diff(lift_indices) > 1) + 1)
        for run in lift_runs:
            if len(run):
                x0 = x_position(int(run[0]))
                x1 = x_position(int(run[-1]))
                draw.rectangle((x0, plot[1], max(x0 + 1, x1), plot[3]), fill="#e9ecef")

    for tick in range(5):
        value = tick / 4
        y = y_position(value)
        draw.line((plot[0], y, plot[2], y), fill="#dce1e5", width=1)
        draw.text((48, y - 8), f"{value:.2f}", fill="#50575e", font=load_font(12))

    previous_stroke = 0
    for index, stroke_id in enumerate(teacher["stroke_ids"]):
        if stroke_id and stroke_id != previous_stroke:
            x = x_position(index)
            for y in range(plot[1], plot[3], 10):
                draw.line((x, y, x, min(y + 5, plot[3])), fill="#9aa3aa", width=1)
        previous_stroke = int(stroke_id)

    series = (
        (teacher["target_width_ratios"], "#687681", "Target width ratio"),
        (teacher["actions"][:, 2], "#e28a25", "Teacher command"),
        (teacher["actual_pressures"], "#168f88", "Teacher actual"),
        (policy["actions"][:, 2], "#d44343", "BC command"),
        (policy["actual_pressures"], "#2769a6", "BC actual"),
    )
    for values, color, _label in series:
        points = [(x_position(index), y_position(value)) for index, value in enumerate(values)]
        draw.line(points, fill=color, width=3)

    tick_count = min(9, step_count)
    for index in np.linspace(0, step_count - 1, tick_count, dtype=int):
        x = x_position(int(index))
        draw.line((x, plot[3], x, plot[3] + 5), fill="#7d858d", width=1)
        draw.text(
            (x - 14, plot[3] + 10),
            f"{index / control_hz:.1f}",
            fill="#50575e",
            font=load_font(11),
        )
    draw.rectangle(plot, outline="#7d858d", width=1)
    draw.text((560, plot[3] + 35), "Elapsed time (s)", fill="#50575e", font=load_font(13))
    draw.text(
        (100, 495),
        "Normalized pressure / target-width ratio (0-1)",
        fill="#50575e",
        font=load_font(12),
    )
    legend_y = 520
    for index, (_values, color, label) in enumerate(series):
        x = 100 + (index % 3) * 360
        y = legend_y + (index // 3) * 24
        draw.line((x, y + 8, x + 24, y + 8), fill=color, width=4)
        draw.text((x + 32, y), label, fill="#343a40", font=load_font(12))
    draw.rectangle((820, legend_y + 26, 840, legend_y + 40), fill="#e9ecef")
    draw.text((848, legend_y + 24), "Brush lifted", fill="#343a40", font=load_font(12))
    draw.text(
        (100, 585),
        f"{control_hz:.0f} Hz | pressure slew limit {pressure_rate:.1f}/s | dashed lines mark stroke starts",
        fill="#59636c",
        font=load_font(12),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_pressure_diagnostics(data_dir, checkpoint_path, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_observation_checkpoint(checkpoint)
    characters = tuple(checkpoint.get("characters", TRAIN_CHARACTERS))
    if not characters or any(character not in TRAIN_CHARACTERS for character in characters):
        raise ValueError("Pressure diagnostics only support the BC training characters")

    policy = BehaviorCloningPolicy(
        observation_size=checkpoint.get("observation_size", OBSERVATION_SIZE),
        hidden_size=checkpoint.get("hidden_size", POLICY_HIDDEN_SIZE),
        action_size=checkpoint.get("action_size", ACTION_SIZE),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    env_kwargs = {
        "max_xy_offset_px": checkpoint.get("max_xy_offset_px", MAX_XY_OFFSET_PX),
        "control_hz": checkpoint.get("control_hz", CONTROL_HZ),
        "max_pressure_rate_per_second": checkpoint.get(
            "max_pressure_rate_per_second", MAX_PRESSURE_RATE_PER_SECOND
        ),
    }
    env = BrushCalligraphyEnv(artifact_dir=data_dir, characters=characters, **env_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for character in characters:
        teacher = rollout_full_character(env, character)
        policy_rollout = rollout_full_character(env, character, policy)
        if len(teacher["trajectory"]) != len(policy_rollout["trajectory"]):
            raise RuntimeError(f"Teacher and BC rollout lengths differ for {character}")
        graph_path = output_dir / f"pressure_all_{character}.png"
        csv_path = output_dir / f"pressure_all_{character}.csv"
        create_full_pressure_plot(
            character,
            teacher,
            policy_rollout,
            env.control_hz,
            env.max_pressure_rate_per_second,
            graph_path,
        )
        save_full_pressure_csv(teacher, policy_rollout, env.control_hz, csv_path)
        output_paths.extend((graph_path, csv_path))
    return output_paths


def create_diagnostic(artifact_dir, checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_observation_checkpoint(checkpoint)
    policy = BehaviorCloningPolicy(
        observation_size=checkpoint.get("observation_size", OBSERVATION_SIZE),
        hidden_size=checkpoint.get("hidden_size", POLICY_HIDDEN_SIZE),
        action_size=checkpoint.get("action_size", ACTION_SIZE),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    target = Image.open(artifact_dir / "target.png").convert("L")
    target_mask = np.asarray(target) < 128
    env_kwargs = {
        "max_xy_offset_px": checkpoint.get("max_xy_offset_px", MAX_XY_OFFSET_PX),
        "control_hz": checkpoint.get("control_hz", CONTROL_HZ),
        "max_pressure_rate_per_second": checkpoint.get(
            "max_pressure_rate_per_second", MAX_PRESSURE_RATE_PER_SECOND
        ),
    }
    teacher_trace, teacher_actions, teacher_positions, teacher_actual, teacher_env = rollout(
        artifact_dir, env_kwargs=env_kwargs
    )
    policy_trace, policy_actions, policy_positions, policy_actual, _ = rollout(
        artifact_dir, policy, env_kwargs
    )
    trajectory = teacher_env.trajectory

    canvas = Image.new("RGB", (1440, 1060), "#f7f8f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 20), "BC endpoint diagnosis", fill="#171b1f", font=load_font(30))
    draw.text(
        (40, 58),
        "Target taper, rule-teacher rollout, and learned-policy rollout",
        fill="#59636c",
        font=load_font(16),
    )

    add_full_view(canvas, draw, "Target", target, 35, 100)
    add_full_view(canvas, draw, "Rule teacher", teacher_trace, 490, 100)
    add_full_view(canvas, draw, "BC policy", policy_trace, 945, 100)

    draw.line((35, 485, 1405, 485), fill="#d4d9dd", width=1)
    draw.text((40, 505), "Endpoint detail", fill="#171b1f", font=load_font(24))
    draw.text(
        (40, 536),
        f"Gray = target-only, dark = overlap, orange = overdraw; cyan numbers = final 10 of {len(trajectory)} executed points",
        fill="#59636c",
        font=load_font(14),
    )

    crop = (170, 102, 212, 132)
    add_zoom_panel(
        canvas, draw, "Rule teacher vs target", teacher_trace, target_mask, crop, teacher_positions, 40, 580
    )
    add_zoom_panel(
        canvas, draw, "BC policy vs target", policy_trace, target_mask, crop, policy_positions, 490, 580
    )
    add_pressure_plot(
        canvas,
        draw,
        "Pressure over the full stroke",
        trajectory,
        target_mask,
        teacher_actions,
        teacher_actual,
        policy_actions,
        policy_actual,
        teacher_env,
        940,
        580,
    )
    draw.text(
        (40, 1005),
        f"Brush width: {MIN_BRUSH_WIDTH_PX} to {MAX_BRUSH_WIDTH_PX} px; pressure slew <= {teacher_env.max_pressure_rate_per_second:.1f}/s ({teacher_env.max_pressure_delta:.2f}/step).",
        fill="#59636c",
        font=load_font(13),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARTIFACT_DIR / "bc_policy.pt")
    parser.add_argument("--pressure-only", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--pressure-output-dir",
        type=Path,
        default=DEFAULT_MULTICHAR_ARTIFACT_DIR / "pressure_graphs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "bc_endpoint_diagnostic.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pressure_only:
        for output_path in create_pressure_diagnostics(
            args.data_dir, args.checkpoint, args.pressure_output_dir
        ):
            print(output_path)
    else:
        create_diagnostic(args.artifact_dir, args.checkpoint, args.output)
        print(args.output)