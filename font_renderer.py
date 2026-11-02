#!/usr/bin/env python3
"""Render a one-stroke baseline and compare it with a font-derived target."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from calligraphy_config import (
    CANVAS_SIZE,
    DEFAULT_BASELINE_BRUSH_WIDTH_PX,
    DEFAULT_CHARACTER,
    DEFAULT_FONT_SIZE_PX,
    DEFAULT_TRAJECTORY_STEPS,
    FONT_RENDERER_OUTPUT_DIR,
)


DEFAULT_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\HGRSKP.TTF",   # HG正楷書体-PRO(こっちが本命)
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",  # wslで動く場合の互換用。ただし明朝体。
)


def resolve_default_font():
    return next((font for font in DEFAULT_FONT_CANDIDATES if Path(font).is_file()), None)


def make_target(font_path, font_size, character):
    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((CANVAS_SIZE / 2, CANVAS_SIZE / 2), character, font=font, fill=0, anchor="mm")
    return image.point(lambda value: 0 if value < 192 else 255)


def make_baseline(target, steps, brush_width):
    bounds = target.point(lambda value: 255 if value == 0 else 0).getbbox()
    if bounds is None:
        raise ValueError("The selected font did not produce a visible glyph.")

    left, _, right, _ = bounds
    row_counts = [sum(target.getpixel((x, y)) == 0 for x in range(target.width)) for y in range(target.height)]
    center_y = max(range(target.height), key=row_counts.__getitem__)
    margin = brush_width / 2
    start_x = max(margin, left)
    end_x = min(CANVAS_SIZE - margin - 1, right - 1)
    if end_x <= start_x:
        raise ValueError("The target glyph is too narrow for the selected brush width.")

    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    draw = ImageDraw.Draw(image)
    trajectory = []
    for index in range(steps):
        progress = index / (steps - 1)
        x = start_x + (end_x - start_x) * progress
        y = center_y
        trajectory.append((progress, x, y, 0.65))
    draw.line([(x, y) for _, x, y, _ in trajectory], fill=0, width=brush_width, joint="curve")
    return image, trajectory


def score(target, prediction):
    target_mask = target.point(lambda value: 1 if value == 0 else 0)
    prediction_mask = prediction.point(lambda value: 1 if value == 0 else 0)
    target_pixels = list(target_mask.getdata())
    prediction_pixels = list(prediction_mask.getdata())
    intersection = sum(a & b for a, b in zip(target_pixels, prediction_pixels))
    target_count = sum(target_pixels)
    prediction_count = sum(prediction_pixels)
    union = target_count + prediction_count - intersection
    return {
        "dice": 2 * intersection / (target_count + prediction_count),
        "iou": intersection / union,
        "overflow_rate": (prediction_count - intersection) / prediction_count,
        "unpainted_rate": (target_count - intersection) / target_count,
        "target_pixels": target_count,
        "prediction_pixels": prediction_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="TTF/TTC font file used for the target image")
    parser.add_argument("--character", default=DEFAULT_CHARACTER, help="Single character rendered as the target glyph")
    parser.add_argument(
        "--font-px", dest="font_size", type=int, default=DEFAULT_FONT_SIZE_PX, help="Font size in pixels"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAJECTORY_STEPS,
        help="Number of trajectory sample points, including both endpoints (steps - 1 intervals)",
    )
    parser.add_argument(
        "--brush-px",
        dest="brush_width",
        type=int,
        default=DEFAULT_BASELINE_BRUSH_WIDTH_PX,
        help="Brush width in pixels",
    )
    parser.add_argument("--output-dir", type=Path, default=FONT_RENDERER_OUTPUT_DIR)
    args = parser.parse_args()

    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if len(args.character) != 1:
        parser.error("--character must be exactly one character")
    if args.font_size < 1 or args.brush_width < 1:
        parser.error("--font-px and --brush-px must be positive")

    font_path = args.font or resolve_default_font()
    if font_path is None:
        parser.error("No default font found; pass a font file with --font")

    target = make_target(font_path, args.font_size, args.character)
    baseline, trajectory = make_baseline(target, args.steps, args.brush_width)
    metrics = score(target, baseline)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target.save(args.output_dir / "target.png")
    baseline.save(args.output_dir / "baseline.png")
    with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("progress", "x_px", "y_px", "z_press"))
        writer.writerows(trajectory)
    result = {
        "character": args.character,
        "font_path": str(Path(font_path).resolve()),
        "font_size": args.font_size,
        "brush_width_px": args.brush_width,
        "steps": args.steps,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()