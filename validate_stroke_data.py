#!/usr/bin/env python3
"""Validate character CSV/PNG pairs and render training-character baselines."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brush_env import BrushCalligraphyEnv
from calligraphy_config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DATA_VALIDATION_DIR,
    HELD_OUT_CHARACTER,
    TRAIN_CHARACTERS,
)


CHARACTERS = TRAIN_CHARACTERS + (HELD_OUT_CHARACTER,)
STROKE_COLORS = ((220, 55, 45, 220), (24, 130, 190, 220), (20, 155, 105, 220))


def read_rows(data_dir, character):
    with (data_dir / f"{character}.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_overlay(target, rows, path):
    base = target.convert("RGBA")
    strokes = {}
    for row in rows:
        strokes.setdefault(int(row["stroke_id"]), []).append(
            (round(float(row["x_px"])), round(float(row["y_px"])))
        )
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for index, points in enumerate(strokes.values()):
        color = STROKE_COLORS[index % len(STROKE_COLORS)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    Image.alpha_composite(base, layer).convert("RGB").save(path)


def render_baseline(env, character, path):
    env.reset(options={"character": character})
    terminated = False
    info = {}
    while not terminated:
        _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 0.65], dtype=np.float32))
    trace = np.asarray(env.render())[:, :, 0] < 128
    target = env.target_mask
    intersection = int((trace & target).sum())
    trace_count = int(trace.sum())
    target_count = int(target.sum())
    union = int((trace | target).sum())
    Image.fromarray(env.render()).convert("L").save(path)
    return {
        "dice": 2 * intersection / max(trace_count + target_count, 1),
        "iou": intersection / max(union, 1),
        "overflow_rate": (trace_count - intersection) / max(trace_count, 1),
        "unpainted_rate": (target_count - intersection) / max(target_count, 1),
        "steps": len(env.trajectory),
        "lift_steps": int((~env.contact).sum()),
    }


def validate(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_summary = {}
    baseline_summary = {}
    for character in CHARACTERS:
        env = BrushCalligraphyEnv(artifact_dir=data_dir, characters=(character,))
        data = env.datasets[character]
        rows = read_rows(data_dir, character)
        points = np.asarray(
            [[float(row["x_px"]), float(row["y_px"])] for row in rows],
            dtype=np.float32,
        )
        stroke_ids = [int(row["stroke_id"]) for row in rows]
        asset_summary[character] = {
            "image_size": list(data["target"].size),
            "rows": len(rows),
            "strokes": len(set(stroke_ids)),
            "points_per_stroke": [stroke_ids.count(stroke_id) for stroke_id in dict.fromkeys(stroke_ids)],
            "xy_bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        }
        save_overlay(data["target"], rows, output_dir / f"overlay_{character}.png")
        if character in TRAIN_CHARACTERS:
            baseline_summary[character] = render_baseline(
                env, character, output_dir / f"baseline_{character}.png"
            )

    summary = {
        "assets": asset_summary,
        "baseline_training_characters": baseline_summary,
        "held_out_character": HELD_OUT_CHARACTER,
        "held_out_used_for_baseline_scoring": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_VALIDATION_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(**vars(parse_args())), indent=2))