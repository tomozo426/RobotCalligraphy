#!/usr/bin/env python3
"""Create final image and trajectory summaries for all four characters."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from brush_env import BrushCalligraphyEnv
from calligraphy_config import DEFAULT_DATA_DIR, HELD_OUT_CHARACTER, PROJECT_ROOT, TRAIN_CHARACTERS
from evaluate_heldout import (
    load_bc_policy,
    load_rl_policy,
    render_trajectory_overlay,
    rollout,
    save_trajectory_overlays,
    summary_font,
)


CHARACTERS = TRAIN_CHARACTERS + (HELD_OUT_CHARACTER,)
METHODS = (
    ("baseline", "Baseline"),
    ("bc", "BC"),
    ("bc_rl", "BC + RL"),
)


def save_final_image_grid(targets, images, output_path):
    labels = ("Target", "Baseline", "BC", "BC + RL")
    cell_size = 220
    left = 58
    top = 44
    margin = 12
    row_height = cell_size + 30
    canvas = Image.new(
        "RGB",
        (left + len(labels) * (cell_size + margin), top + len(CHARACTERS) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        x = left + column * (cell_size + margin)
        draw.text((x, 8), label, fill="#20252b", font=summary_font(22))
    for row, character in enumerate(CHARACTERS):
        y = top + row * row_height
        draw.text((8, y + cell_size // 2 - 12), character, fill="#20252b", font=summary_font(24))
        row_images = [targets[character]] + [images[(character, key)] for key, _ in METHODS]
        for column, image in enumerate(row_images):
            x = left + column * (cell_size + margin)
            canvas.paste(image.convert("RGB").resize((cell_size, cell_size)), (x, y))
            draw.rectangle(
                (x, y, x + cell_size - 1, y + cell_size - 1),
                outline="#9aa2a9",
            )
    canvas.save(output_path)


def save_trajectory_grid(targets, trajectories, output_path):
    margin = 12
    panel_width = 512
    panel_height = 586
    canvas = Image.new(
        "RGB",
        (len(METHODS) * panel_width + (len(METHODS) + 1) * margin,
         len(CHARACTERS) * panel_height + (len(CHARACTERS) + 1) * margin),
        "white",
    )
    for row, character in enumerate(CHARACTERS):
        for column, (method, label) in enumerate(METHODS):
            panel = render_trajectory_overlay(
                targets[character], trajectories[(character, method)],
                f"{character} / {label}",
            )
            x = margin + column * (panel_width + margin)
            y = margin + row * (panel_height + margin)
            canvas.paste(panel, (x, y))
    canvas.save(output_path)


def main(args):
    bc_checkpoint, bc_policy = load_bc_policy(args.bc_checkpoint)
    _, rl_policy = load_rl_policy(args.rl_checkpoint, bc_checkpoint)
    env = BrushCalligraphyEnv(
        artifact_dir=args.data_dir,
        characters=CHARACTERS,
        lock_full_pressure=bc_checkpoint.get("lock_full_pressure", 0.5),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        character: env.datasets[character]["target"].convert("L")
        for character in CHARACTERS
    }
    images = {}
    trajectories = {}
    results = {}
    combined_rows = []
    for character in CHARACTERS:
        results[character] = {}
        character_trajectories = {}
        for method, _label in METHODS:
            policy = {"baseline": None, "bc": bc_policy, "bc_rl": rl_policy}[method]
            image, metrics, rows = rollout(env, method, policy, character=character)
            images[(character, method)] = image
            trajectories[(character, method)] = rows
            character_trajectories[method] = rows
            results[character][method] = metrics
            image.save(args.output_dir / f"{method}_{character}.png")
            combined_rows.extend(
                {"character": character, "method": method, **trace_row}
                for trace_row in rows
            )
        save_trajectory_overlays(
            targets[character], character_trajectories, args.output_dir,
            character=character,
        )

    save_final_image_grid(
        targets, images, args.output_dir / "final_images_all_characters.png"
    )
    save_trajectory_grid(
        targets, trajectories, args.output_dir / "trajectories_all_characters.png"
    )
    with (args.output_dir / "trajectories_all_characters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        trace_fields = [
            key for key in combined_rows[0] if key not in {"character", "method"}
        ]
        writer = csv.DictWriter(
            file, fieldnames=["character", "method", *trace_fields]
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    summary = {
        "characters": list(CHARACTERS),
        "training_characters": list(TRAIN_CHARACTERS),
        "held_out_character": HELD_OUT_CHARACTER,
        "held_out_data_used_for_training_or_tuning": False,
        "models_updated": False,
        "baseline": "center-corrected reference with fixed brush width",
        "trajectory_reference": "target cross-section center offset, limited to max XY offset",
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved four-character summary to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=PROJECT_ROOT / "artifacts" / "final_summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())