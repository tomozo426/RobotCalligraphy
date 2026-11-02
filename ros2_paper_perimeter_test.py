#!/usr/bin/env python3
"""Trace the paper perimeter once lifted and once at paper height."""

import argparse
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import rclpy
from sensor_msgs.msg import Image as RosImage

from calligraphy_config import CANVAS_SIZE, DEFAULT_BASELINE_BRUSH_WIDTH_PX
from ros2_policy_runner import GakuPolicyRunner, base_to_pixel


PAPER_SIZE_M = 0.1024


def build_perimeter(center_x, center_y, margin_m, spacing_m, start_corner):
    half_extent = PAPER_SIZE_M / 2.0 - margin_m
    if half_extent <= 0.0:
        raise ValueError("margin must be smaller than half the paper width")

    corners = {
        "top-right": (center_x + half_extent, center_y + half_extent),
        "top-left": (center_x - half_extent, center_y + half_extent),
        "bottom-left": (center_x - half_extent, center_y - half_extent),
        "bottom-right": (center_x + half_extent, center_y - half_extent),
    }
    names = ("top-right", "top-left", "bottom-left", "bottom-right")
    start_index = names.index(start_corner)
    ordered = [corners[names[(start_index + offset) % len(names)]] for offset in range(4)]
    ordered.append(ordered[0])

    points = []
    for start, end in zip(ordered[:-1], ordered[1:]):
        distance = math.dist(start, end)
        segments = max(1, int(math.ceil(distance / spacing_m)))
        for segment in range(1, segments + 1):
            fraction = segment / segments
            points.append((
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            ))
    return ordered[0], points


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-center-x", type=float, default=0.19)
    parser.add_argument("--paper-center-y", type=float, default=0.0)
    parser.add_argument("--paper-surface-z", type=float, default=0.0005)
    parser.add_argument("--lift-z", type=float, default=0.0205)
    parser.add_argument("--margin-mm", type=float, default=10.0)
    parser.add_argument("--spacing-mm", type=float, default=5.0)
    parser.add_argument("--speed-mm-s", type=float, default=15.0)
    parser.add_argument("--timeout-scale", type=float, default=6.0)
    parser.add_argument("--ik-joint-tolerance-rad", type=float, default=0.30)
    parser.add_argument(
        "--start-corner",
        choices=("top-right", "top-left", "bottom-left", "bottom-right"),
        default="top-right",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-only", action="store_true")
    parser.add_argument("--draw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.margin_mm < 0.0:
        raise ValueError("--margin-mm must be non-negative")
    if (
        args.spacing_mm <= 0.0
        or args.speed_mm_s <= 0.0
        or args.timeout_scale < 1.0
        or args.ik_joint_tolerance_rad <= 0.0
    ):
        raise ValueError("spacing and speed must be positive; timeout-scale must be at least 1; IK joint tolerance must be positive")
    if args.lift_z <= args.paper_surface_z:
        raise ValueError("--lift-z must be above --paper-surface-z")

    margin_m = args.margin_mm / 1000.0
    spacing_m = args.spacing_mm / 1000.0
    speed_m_s = args.speed_mm_s / 1000.0
    start, lap_points = build_perimeter(
        args.paper_center_x,
        args.paper_center_y,
        margin_m,
        spacing_m,
        args.start_corner,
    )
    print(
        f"start={args.start_corner} at ({start[0]:.4f}, {start[1]:.4f}) m; "
        f"{len(lap_points)} points/lap, 1 lifted lap + 1 paper-height lap"
    )
    print(
        f"paper_z={args.paper_surface_z:.4f} m, lift_z={args.lift_z:.4f} m, "
        f"edge margin={args.margin_mm:.1f} mm, speed={args.speed_mm_s:.1f} mm/s, "
        f"draw={'on' if args.draw else 'off'}"
    )
    if args.dry_run:
        return

    rclpy.init()
    runner = None
    try:
        runner = GakuPolicyRunner(
            SimpleNamespace(
                texture_topic="/paper/texture_update" if args.draw else ""
            ),
            env=None,
            policy=None,
            policy_name="baseline",
            checkpoint={},
        )
        canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        canvas_draw = ImageDraw.Draw(canvas) if args.draw else None
        joints = runner.joint_positions()
        current_pose = runner.fk(joints)
        previous_xy = (float(current_pose[0]), float(current_pose[1]))

        def move_to(point, z, duration, constrain_ik=True):
            nonlocal joints, previous_xy
            target = [point[0], point[1], z]
            joint_tolerance = args.ik_joint_tolerance_rad if constrain_ik else None
            target_joints = runner.ik(target, joints, joint_tolerance=joint_tolerance)
            runner.send_joint_goal(
                target_joints,
                duration=duration,
                timeout=max(15.0, duration * args.timeout_scale + 5.0),
            )
            joints = runner.joint_positions()
            previous_xy = point

        def draw_segment(start_point, end_point):
            if canvas_draw is None:
                return
            pixels = [
                base_to_pixel(
                    point[0],
                    point[1],
                    args.paper_center_x,
                    args.paper_center_y,
                    PAPER_SIZE_M / CANVAS_SIZE,
                )
                for point in (start_point, end_point)
            ]
            start_pixel, end_pixel = [
                tuple(int(round(value)) for value in pixel) for pixel in pixels
            ]
            brush_radius = DEFAULT_BASELINE_BRUSH_WIDTH_PX // 2
            canvas_draw.line(
                (start_pixel, end_pixel),
                fill=(16, 16, 16),
                width=DEFAULT_BASELINE_BRUSH_WIDTH_PX,
            )
            canvas_draw.ellipse(
                (
                    end_pixel[0] - brush_radius,
                    end_pixel[1] - brush_radius,
                    end_pixel[0] + brush_radius,
                    end_pixel[1] + brush_radius,
                ),
                fill=(16, 16, 16),
            )
            pixels = np.asarray(canvas, dtype=np.uint8)
            message = RosImage()
            message.header.stamp = runner.get_clock().now().to_msg()
            message.header.frame_id = "paper"
            message.height, message.width = pixels.shape[:2]
            message.encoding = "rgb8"
            message.step = message.width * 3
            message.data = pixels.tobytes()
            runner.texture_publisher.publish(message)

        if current_pose[2] < args.lift_z:
            move_to(
                previous_xy,
                args.lift_z,
                max(5.0, (args.lift_z - current_pose[2]) / speed_m_s),
                constrain_ik=False,
            )
        horizontal_distance = math.dist(previous_xy, start)
        move_to(
            start,
            args.lift_z,
            max(5.0, horizontal_distance / speed_m_s),
            constrain_ik=False,
        )
        if args.start_only:
            print(f"Reached perimeter start at lift z={args.lift_z:.4f} m")
            return

        print("Starting one lifted perimeter lap")
        for index, point in enumerate(lap_points, start=1):
            distance = math.dist(previous_xy, point)
            move_to(point, args.lift_z, max(0.2, distance / speed_m_s))
            if index % 20 == 0:
                print(f"lifted lap: {index}/{len(lap_points)}")

        print("Lowering to paper height for one lap")
        move_to(start, args.paper_surface_z, 1.0)
        for lap in range(1, 2):
            for index, point in enumerate(lap_points, start=1):
                distance = math.dist(previous_xy, point)
                segment_start = previous_xy
                move_to(point, args.paper_surface_z, max(0.2, distance / speed_m_s))
                draw_segment(segment_start, point)
                if index % 20 == 0:
                    print(f"paper-height lap {lap}/1: {index}/{len(lap_points)}")

        print("Lifting at the perimeter start")
        move_to(start, args.lift_z, 1.0)
        print("Perimeter test completed")
    finally:
        if runner is not None:
            runner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()