#!/usr/bin/env python3
"""Gymnasium environment for tracing multi-stroke calligraphy data."""

import csv
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from gymnasium import spaces

from ros2_feedback import MotionUnreachableError
from calligraphy_config import (
    ACTION_SIZE,
    BRUSH_DAB_CAP_SEGMENTS,
    BRUSH_DAB_CAP_LENGTH_RATIO,
    BRUSH_DAB_SIDE_SEGMENTS,
    BRUSH_DIRECTION_LOCK_START_PRESSURE,
    CANVAS_SIZE,
    CONTROL_HZ,
    CROSSING_EXCLUSION_RADIUS_PX,
    CROSSING_MAX_TANGENT_ALIGNMENT,
    DEFAULT_DATA_DIR,
    LIFT_SPEED_PX_PER_SECOND,
    MAX_BRUSH_WIDTH_PX,
    MAX_BRUSH_TURN_DEGREES_PER_STEP,
    MAX_PRESSURE_RATE_PER_SECOND,
    MAX_XY_OFFSET_PX,
    MIN_BRUSH_WIDTH_PX,
    OBSERVATION_SIZE,
    TARGET_SPEED_PX_PER_SECOND,
    TRAIN_CHARACTERS,
)


OBSERVATION_CONVENTION = "target_width_pixels_over_canvas_span_v1"


def validate_observation_checkpoint(checkpoint):
    if checkpoint.get("observation_convention") != OBSERVATION_CONVENTION:
        raise ValueError(
            "Checkpoint uses an incompatible observation convention; "
            "retrain BC and PPO with normalized target width observations"
        )


def width_from_pressure(pressure, max_brush_width_px=MAX_BRUSH_WIDTH_PX):
    pressure = float(np.clip(pressure, 0.0, 1.0))
    return MIN_BRUSH_WIDTH_PX + round(
        pressure * (max_brush_width_px - MIN_BRUSH_WIDTH_PX)
    )


def pressure_from_width(width_px, max_brush_width_px=MAX_BRUSH_WIDTH_PX):
    return float(
        np.clip(
            (float(width_px) - MIN_BRUSH_WIDTH_PX)
            / (max_brush_width_px - MIN_BRUSH_WIDTH_PX),
            0.0,
            1.0,
        )
    )


def pressure_to_z(pressure, contact_z_m, lift_z_m):
    if lift_z_m <= contact_z_m:
        raise ValueError("lift_z_m must be greater than contact_z_m")
    pressure = float(np.clip(pressure, 0.0, 1.0))
    return lift_z_m - pressure * (lift_z_m - contact_z_m)


def z_to_pressure(z_m, contact_z_m, lift_z_m):
    if lift_z_m <= contact_z_m:
        raise ValueError("lift_z_m must be greater than contact_z_m")
    return float(
        np.clip((lift_z_m - z_m) / (lift_z_m - contact_z_m), 0.0, 1.0)
    )


def contact_pressure_to_z(pressure, contact_z_m, pressure_z_span_m):
    if pressure_z_span_m <= 0.0:
        raise ValueError("pressure_z_span_m must be positive")
    pressure = float(np.clip(pressure, 0.0, 1.0))
    return float(contact_z_m) - pressure * float(pressure_z_span_m)


def z_to_contact_pressure(z_m, contact_z_m, pressure_z_span_m):
    if pressure_z_span_m <= 0.0:
        raise ValueError("pressure_z_span_m must be positive")
    return float(
        np.clip(
            (float(contact_z_m) - float(z_m)) / float(pressure_z_span_m),
            0.0,
            1.0,
        )
    )


class BrushCalligraphyEnv(gym.Env):
    """Trace one selected character, lifting the brush between strokes."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(CONTROL_HZ)}

    def __init__(
        self,
        artifact_dir=DEFAULT_DATA_DIR,
        characters=TRAIN_CHARACTERS,
        max_xy_offset_px=MAX_XY_OFFSET_PX,
        control_hz=CONTROL_HZ,
        max_pressure_rate_per_second=MAX_PRESSURE_RATE_PER_SECOND,
        target_speed_px_per_second=TARGET_SPEED_PX_PER_SECOND,
        lift_speed_px_per_second=LIFT_SPEED_PX_PER_SECOND,
        lock_full_pressure=0.5,
        endpoint_weight=1.0,
        path_weight=1.0,
        render_mode=None,
        motion_backend=None,
    ):
        super().__init__()
        self.endpoint_weight = float(endpoint_weight)
        self.path_weight = float(path_weight)
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if min(control_hz, max_pressure_rate_per_second, target_speed_px_per_second, lift_speed_px_per_second) <= 0:
            raise ValueError("control rates and speeds must be positive")
        self.render_mode = render_mode
        self.motion_backend = motion_backend
        self.max_xy_offset_px = float(max_xy_offset_px)
        self.control_hz = float(control_hz)
        self.max_pressure_rate_per_second = float(max_pressure_rate_per_second)
        self.max_pressure_delta = self.max_pressure_rate_per_second / self.control_hz
        self.lock_full_pressure = float(lock_full_pressure)
        if not BRUSH_DIRECTION_LOCK_START_PRESSURE < self.lock_full_pressure <= 1.0:
            raise ValueError(
                "lock_full_pressure must be greater than "
                f"{BRUSH_DIRECTION_LOCK_START_PRESSURE} and at most 1.0"
            )
        self.target_speed_px_per_second = float(target_speed_px_per_second)
        self.lift_speed_px_per_second = float(lift_speed_px_per_second)
        self.max_brush_width_px = MAX_BRUSH_WIDTH_PX

        artifact_dir = Path(artifact_dir)
        if (artifact_dir / "target.png").is_file():
            dataset_names = {"custom": artifact_dir}
        else:
            dataset_names = {name: artifact_dir for name in characters}
        if not dataset_names:
            raise ValueError("At least one character dataset is required")
        self.datasets = {
            name: self._load_dataset(name, directory)
            for name, directory in dataset_names.items()
        }
        self.character_names = tuple(self.datasets)
        self.character = self.character_names[0]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.ones(ACTION_SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        observation_low = np.array(
            [0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0], dtype=np.float32
        )
        observation_high = np.ones(OBSERVATION_SIZE, dtype=np.float32)
        if self.motion_backend is not None:
            observation_low[[0, 1, 4, 5]] = -np.inf
            observation_high[[0, 1, 4, 5]] = np.inf
        self.observation_space = spaces.Box(
            low=observation_low, high=observation_high, dtype=np.float32
        )
        self.step_index = 0
        self.position = np.zeros(2, dtype=np.float32)
        self.pressure = 0.0
        self._brush_heading = None
        self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.canvas = None
        self._previous_position = np.zeros(2, dtype=np.float32)
        self._select_dataset(self.character)

    def _load_dataset(self, name, directory):
        directory = Path(directory)
        csv_path = directory / (name + ".csv") if name != "custom" else directory / "trajectory.csv"
        image_path = directory / (name + ".png") if name != "custom" else directory / "target.png"
        if not csv_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"Missing CSV/PNG pair for {name}: {csv_path}, {image_path}")
        target = Image.open(image_path).convert("L")
        if target.size != (CANVAS_SIZE, CANVAS_SIZE):
            raise ValueError(f"{image_path} must be {CANVAS_SIZE}x{CANVAS_SIZE}")
        if not (np.asarray(target) < 128).any():
            raise ValueError(f"{image_path} contains no target ink")

        with csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"progress", "x_px", "y_px"}
            if name != "custom":
                required.add("stroke_id")
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{csv_path} must contain columns {sorted(required)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{csv_path} contains no trajectory points")

        strokes = []
        current_id = None
        for row_index, row in enumerate(rows):
            stroke_id = int(row.get("stroke_id") or 1)
            point = np.asarray(
                [float(row["x_px"]), float(row["y_px"])],
                dtype=np.float32,
            )
            progress = float(row.get("progress", row_index / max(len(rows) - 1, 1)))
            if not np.isfinite(point).all() or not math.isfinite(progress):
                raise ValueError(f"{csv_path} contains non-finite values")
            if not (0 <= point[0] < CANVAS_SIZE and 0 <= point[1] < CANVAS_SIZE):
                raise ValueError(f"{csv_path} contains coordinates outside the canvas")
            if not 0 <= progress <= 1:
                raise ValueError(f"{csv_path} progress must be in [0, 1]")
            if stroke_id != current_id:
                if any(stroke_id == previous_id for previous_id, _ in strokes):
                    raise ValueError(f"{csv_path} stroke IDs must be contiguous")
                strokes.append((stroke_id, []))
                current_id = stroke_id
            strokes[-1][1].append((progress, point))

        for _stroke_id, points in strokes:
            progress = [item[0] for item in points]
            if any(right < left for left, right in zip(progress, progress[1:])):
                raise ValueError(f"{csv_path} progress must not decrease within a stroke")
        stroke_paths = {
            stroke_id: np.asarray([point[:2] for _, point in points], dtype=np.float32)
            for stroke_id, points in strokes
        }

        spacing = self.target_speed_px_per_second / self.control_hz
        sampled_strokes = [
            self._resample_stroke(
                [point for _, point in points],
                [progress for progress, _ in points],
                spacing,
            )
            for _, points in strokes
        ]
        trajectory, contact, stroke_ids, stroke_progress = [], [], [], []
        lift_spacing = self.lift_speed_px_per_second / self.control_hz
        for stroke_index, stroke in enumerate(sampled_strokes):
            if stroke_index:
                start = sampled_strokes[stroke_index - 1][0][-1][:2]
                end = stroke[0][0][:2]
                distance = float(np.linalg.norm(end - start))
                lift_steps = max(1, int(math.ceil(distance / lift_spacing)))
                for fraction in np.linspace(1 / lift_steps, 1.0, lift_steps):
                    trajectory.append(start + fraction * (end - start))
                    contact.append(False)
                    stroke_ids.append(0)
                    stroke_progress.append(0.0)
            for point, progress in zip(*stroke):
                trajectory.append(point)
                contact.append(True)
                stroke_ids.append(strokes[stroke_index][0])
                stroke_progress.append(float(progress))

        return {
            "target": target,
            "target_mask": np.asarray(target) < 128,
            "trajectory": np.asarray(trajectory, dtype=np.float32),
            "contact": np.asarray(contact, dtype=bool),
            "stroke_ids": np.asarray(stroke_ids, dtype=np.int32),
            "stroke_progress": np.asarray(stroke_progress, dtype=np.float32),
            "stroke_paths": stroke_paths,
        }

    @staticmethod
    def _resample_stroke(points, progress, spacing):
        points = np.asarray(points, dtype=np.float32)
        progress = np.asarray(progress, dtype=np.float32)
        if len(points) == 1:
            return points.copy(), progress.copy()
        distances = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(distances)]
        total = float(cumulative[-1])
        if total == 0:
            return points[[0, -1]].copy(), progress[[0, -1]].copy()
        count = max(2, int(math.ceil(total / spacing)) + 1)
        samples = np.linspace(0.0, total, count, dtype=np.float32)
        resampled = np.column_stack(
            [np.interp(samples, cumulative, points[:, axis]) for axis in range(points.shape[1])]
        ).astype(np.float32)
        return resampled, np.interp(samples, cumulative, progress).astype(np.float32)

    def _select_dataset(self, character):
        data = self.datasets[character]
        self.character = character
        self.target = data["target"]
        self.target_mask = data["target_mask"]
        self.target_pixel_count = int(self.target_mask.sum())
        self.trajectory = data["trajectory"]
        self.contact = data["contact"]
        self.stroke_ids = data["stroke_ids"]
        self.stroke_progress = data["stroke_progress"]
        self.stroke_paths = data["stroke_paths"]
        distances = np.linalg.norm(np.diff(self.trajectory[:, :2], axis=0), axis=1)
        self.deviation_scale = self.max_xy_offset_px + float(distances.max(initial=0.0))
        self.target_cross_sections, self.crossing_mask = self._build_target_cross_sections()
        self.step_weights = self._build_step_weights()

    def _build_step_weights(self):
        """Stroke start/end points get endpoint_weight, other contact points path_weight."""
        count = len(self.trajectory)
        weights = np.ones(count, dtype=np.float32)
        weights[self.contact] = self.path_weight
        for index in np.flatnonzero(self.contact):
            continues_back = index > 0 and self.contact[index - 1] and self.stroke_ids[index - 1] == self.stroke_ids[index]
            continues_forward = (
                index + 1 < count and self.contact[index + 1] and self.stroke_ids[index + 1] == self.stroke_ids[index]
            )
            if not continues_back or not continues_forward:
                weights[index] = self.endpoint_weight
        return weights

    @staticmethod
    def _point_segment_distance(point, start, end):
        vector = end - start
        length_squared = float(np.dot(vector, vector))
        if length_squared == 0:
            return float("inf"), np.zeros(2, dtype=np.float32)
        fraction = float(np.clip(np.dot(point - start, vector) / length_squared, 0.0, 1.0))
        nearest = start + fraction * vector
        return float(np.linalg.norm(point - nearest)), vector / np.sqrt(length_squared)

    @classmethod
    def _nearest_path_segment(cls, point, path):
        best_distance = float("inf")
        best_tangent = np.zeros(2, dtype=np.float32)
        if len(path) == 1:
            return float(np.linalg.norm(point - path[0])), best_tangent
        for start, end in zip(path[:-1], path[1:]):
            distance, tangent = cls._point_segment_distance(point, start, end)
            if distance < best_distance:
                best_distance = distance
                best_tangent = tangent
        return best_distance, best_tangent

    def _raw_target_cross_section(self, center, tangent):
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        x0, y0 = np.maximum(np.floor(center - 36), 0).astype(int)
        x1 = min(self.target_mask.shape[1], int(np.ceil(center[0] + 37)))
        y1 = min(self.target_mask.shape[0], int(np.ceil(center[1] + 37)))
        ys, xs = np.nonzero(self.target_mask[y0:y1, x0:x1])
        if not len(xs):
            return 0.0, float(MIN_BRUSH_WIDTH_PX)
        points = np.column_stack((xs + x0, ys + y0)).astype(np.float32)
        relative = points - center
        along = relative @ tangent
        across = relative @ normal
        selected = (np.abs(along) <= 5.0) & (np.abs(across) <= 36.0)
        if not np.any(selected):
            return 0.0, float(MIN_BRUSH_WIDTH_PX)

        occupied = np.zeros(73, dtype=bool)
        bins = np.clip(np.rint(across[selected]).astype(int) + 36, 0, 72)
        occupied[bins] = True
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        intervals = [(int(start) - 36, int(end) - 36) for start, end in zip(starts, ends)]
        if not intervals:
            return 0.0, float(MIN_BRUSH_WIDTH_PX)
        lower, upper = min(intervals, key=lambda interval: max(interval[0], -interval[1], 0))
        return (lower + upper) / 2.0, float(upper - lower + 1)

    def _build_target_cross_sections(self):
        sections = np.zeros((len(self.trajectory), 2), dtype=np.float32)
        sections[:, 1] = MIN_BRUSH_WIDTH_PX
        crossing_mask = np.zeros(len(self.trajectory), dtype=bool)
        for index, reference in enumerate(self.trajectory):
            if not self.contact[index]:
                continue
            tangent = self._tangent(index)
            if not np.any(tangent):
                continue
            sections[index] = self._raw_target_cross_section(reference[:2], tangent)
            stroke_id = int(self.stroke_ids[index])
            current_path = self.stroke_paths[stroke_id]
            local_segment = -1
            if len(current_path) > 1:
                local_segment = min(
                    int(self.stroke_progress[index] * (len(current_path) - 1)),
                    len(current_path) - 2,
                )
            for candidate_id, candidate_path in self.stroke_paths.items():
                for segment_index, (start, end) in enumerate(
                    zip(candidate_path[:-1], candidate_path[1:])
                ):
                    if candidate_id == stroke_id and segment_index == local_segment:
                        continue
                    distance, other_tangent = self._point_segment_distance(
                        reference[:2], start, end
                    )
                    if (
                        distance <= CROSSING_EXCLUSION_RADIUS_PX
                        and np.any(other_tangent)
                        and abs(float(np.dot(tangent, other_tangent)))
                        <= CROSSING_MAX_TANGENT_ALIGNMENT
                    ):
                        crossing_mask[index] = True
                        break
                if crossing_mask[index]:
                    break

        for stroke_id in self.stroke_paths:
            indices = np.flatnonzero(self.contact & (self.stroke_ids == stroke_id))
            crossing_indices = indices[crossing_mask[indices]]
            valid_indices = indices[~crossing_mask[indices]]
            if len(crossing_indices) and len(valid_indices):
                for column in range(sections.shape[1]):
                    sections[crossing_indices, column] = np.interp(
                        crossing_indices, valid_indices, sections[valid_indices, column]
                    )
        return sections, crossing_mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        default_character = self.character_names[int(self.np_random.integers(len(self.character_names)))]
        character = options.get("character", default_character)
        if character not in self.datasets:
            raise ValueError(f"Unknown character {character!r}; choose from {self.character_names}")
        self._select_dataset(character)
        self.step_index = 0
        self.position = self.trajectory[0, :2].copy()
        self.pressure = 0.0
        self.observation_pressure = 0.0
        if self.motion_backend is not None:
            reset_result = self.motion_backend.reset(self.position.copy())
            if isinstance(reset_result, dict):
                self.position = self._validated_position(
                    reset_result["measured_position"], "reset"
                )
                self.observation_pressure = float(
                    np.clip(reset_result.get("measured_pressure", 0.0), 0.0, 1.0)
                )
            else:
                self.position = self._validated_position(reset_result, "reset")
        self._brush_heading = None
        self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._previous_position = self.position.copy()
        self.canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        return self._observation(), {"progress": 0.0, "character": self.character}

    @staticmethod
    def _validated_position(position, operation):
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError(
                f"Motion backend {operation} must return two finite pixel coordinates"
            )
        return position.copy()

    @staticmethod
    def _normalize_vector(vector, fallback):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.asarray(fallback, dtype=np.float32)
        return vector / norm

    def _brush_reference_tangent(self, index):
        if (
            index > 0
            and self.contact[index - 1]
            and self.stroke_ids[index - 1] == self.stroke_ids[index]
        ):
            vector = self.trajectory[index, :2] - self.trajectory[index - 1, :2]
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                return vector / norm
        return self._tangent(index)

    def _brush_tangent(self, reference_tangent, same_stroke_before):
        reference_tangent = self._normalize_vector(reference_tangent, np.array([1.0, 0.0]))
        if not same_stroke_before or self._brush_heading is None:
            self._brush_heading = reference_tangent.copy()
            return self._brush_heading.copy()

        movement = self.position - self._previous_position
        movement_norm = float(np.linalg.norm(movement))
        if movement_norm > 0.0:
            movement = movement / movement_norm
            lock_progress = np.clip(
                (self.pressure - BRUSH_DIRECTION_LOCK_START_PRESSURE)
                / (self.lock_full_pressure - BRUSH_DIRECTION_LOCK_START_PRESSURE),
                0.0,
                1.0,
            )
            lock_progress = lock_progress * lock_progress * (3.0 - 2.0 * lock_progress)
            blended = movement * (1.0 - lock_progress) + reference_tangent * lock_progress
            target_tangent = self._normalize_vector(blended, reference_tangent)
        else:
            target_tangent = reference_tangent

        current_angle = math.atan2(self._brush_heading[1], self._brush_heading[0])
        target_angle = math.atan2(target_tangent[1], target_tangent[0])
        angle_delta = math.atan2(
            math.sin(target_angle - current_angle),
            math.cos(target_angle - current_angle),
        )
        max_turn = math.radians(MAX_BRUSH_TURN_DEGREES_PER_STEP)
        next_angle = current_angle + float(np.clip(angle_delta, -max_turn, max_turn))
        self._brush_heading = np.asarray(
            [math.cos(next_angle), math.sin(next_angle)], dtype=np.float32
        )
        return self._brush_heading.copy()

    def brush_width_for_pressure(self, pressure):
        pressure = float(np.clip(pressure, 0.0, 1.0))
        return width_from_pressure(pressure, self.max_brush_width_px)

    def pressure_for_brush_width(self, width_px):
        return pressure_from_width(width_px, self.max_brush_width_px)

    def _draw_brush_dab(self, draw, position, brush_width, tangent, speed):
        radius = max(float(brush_width) * 0.5, 0.2)
        flexibility = 0.5
        tangent = np.asarray(tangent, dtype=np.float32)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm == 0.0:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        length = radius * (0.85 + 1.6 * flexibility) + min(
            radius * 0.5, max(float(speed), 0.0) * radius * 0.20
        )
        base = np.asarray(position, dtype=np.float32) - tangent * length
        cap_length = radius * BRUSH_DAB_CAP_LENGTH_RATIO
        front_center = np.asarray(position, dtype=np.float32) - tangent * cap_length
        taper_length = length - cap_length
        points = []
        for index in range(BRUSH_DAB_SIDE_SEGMENTS + 1):
            progress = index / BRUSH_DAB_SIDE_SEGMENTS
            width_progress = progress * progress * (3.0 - 2.0 * progress)
            half_width = radius * (0.12 + 0.88 * width_progress)
            center = base + tangent * taper_length * progress
            curve = np.sin(np.pi * progress) ** 2 * radius * 0.08 * flexibility
            points.append(tuple(center + normal * (half_width + curve)))

        for index in range(1, BRUSH_DAB_CAP_SEGMENTS + 1):
            angle = np.pi / 2.0 - np.pi * index / BRUSH_DAB_CAP_SEGMENTS
            cap_point = front_center + radius * (
                tangent * BRUSH_DAB_CAP_LENGTH_RATIO * np.cos(angle)
                + normal * np.sin(angle)
            )
            points.append(tuple(cap_point))

        for index in range(BRUSH_DAB_SIDE_SEGMENTS - 1, -1, -1):
            progress = index / BRUSH_DAB_SIDE_SEGMENTS
            width_progress = progress * progress * (3.0 - 2.0 * progress)
            half_width = radius * (0.12 + 0.88 * width_progress)
            center = base + tangent * taper_length * progress
            curve = np.sin(np.pi * progress) ** 2 * radius * 0.08 * flexibility
            points.append(tuple(center - normal * (half_width - curve)))
        draw.polygon(points, fill=0)

    def step(self, action):
        if self.canvas is None:
            raise RuntimeError("Call reset() before step()")
        if self.step_index >= len(self.trajectory):
            raise RuntimeError("Episode is finished; call reset() before stepping again")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_SIZE,):
            raise ValueError(f"Expected action shape ({ACTION_SIZE},), received {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        reference = self.trajectory[self.step_index, :2]
        in_contact = bool(self.contact[self.step_index])
        same_stroke_before = (
            self.step_index > 0
            and in_contact
            and self.contact[self.step_index - 1]
            and self.stroke_ids[self.step_index - 1] == self.stroke_ids[self.step_index]
        )
        if not in_contact:
            action = np.zeros_like(action)
        elif not same_stroke_before:
            self.previous_action = np.zeros_like(self.previous_action)
        offset = action[:2] * self.max_xy_offset_px if in_contact else np.zeros(2, dtype=np.float32)
        command_position = reference + offset
        target_pressure = float(action[2]) if in_contact else 0.0
        pressure_delta = np.clip(
            target_pressure - self.pressure,
            -self.max_pressure_delta,
            self.max_pressure_delta,
        )
        self.pressure = float(np.clip(self.pressure + pressure_delta, 0.0, 1.0))
        motion_failure = None
        if self.motion_backend is None:
            self.position = command_position
            motion_info = {}
        else:
            try:
                motion_result = self.motion_backend.step(
                    command_position.copy(), self.pressure, in_contact
                )
            except MotionUnreachableError as error:
                motion_failure = str(error)
                self.position = self._previous_position.copy()
                motion_info = {}
            else:
                if isinstance(motion_result, dict):
                    if "measured_position" not in motion_result:
                        raise ValueError(
                            "Motion backend result must include measured_position"
                        )
                    motion_info = dict(motion_result.get("info", {}))
                    measured_position = motion_result["measured_position"]
                    self.observation_pressure = float(
                        np.clip(
                            motion_info.get("measured_pressure", self.pressure),
                            0.0,
                            1.0,
                        )
                    )
                else:
                    motion_info = {}
                    measured_position = motion_result
                    self.observation_pressure = self.pressure
                self.position = self._validated_position(measured_position, "step")
        if self.motion_backend is None:
            self.observation_pressure = self.pressure
        render_pressure = (
            self.observation_pressure if self.motion_backend is not None else self.pressure
        )
        brush_width = self.brush_width_for_pressure(render_pressure)

        before = np.asarray(self.canvas) < 128
        if in_contact and motion_failure is None:
            draw = ImageDraw.Draw(self.canvas)
            previous_position = getattr(self, "_previous_position", self.position)
            speed = float(np.linalg.norm(self.position - previous_position))
            self._draw_brush_dab(
                draw,
                self.position,
                brush_width,
                self._brush_tangent(
                    self._brush_reference_tangent(self.step_index), same_stroke_before
                ),
                speed,
            )

        painted = np.asarray(self.canvas) < 128
        newly_painted = painted & ~before
        step_weight = float(self.step_weights[self.step_index])
        coverage_gain = step_weight * float((newly_painted & self.target_mask).sum()) / max(self.target_pixel_count, 1)
        overflow_gain = step_weight * float((newly_painted & ~self.target_mask).sum()) / max(self.target_pixel_count, 1)
        action_cost = float(np.square(action).mean())
        smoothness_cost = float(np.square(action - self.previous_action).mean())
        reward = 5.0 * coverage_gain - 5.0 * overflow_gain - 0.02 * action_cost - 0.02 * smoothness_cost
        if motion_failure is not None:
            reward = -1.0

        self._previous_position = self.position.copy()
        self.previous_action = action.copy()
        self.step_index += 1
        terminated = motion_failure is not None or self.step_index >= len(self.trajectory)
        painted_count = int(painted.sum())
        intersection = int((painted & self.target_mask).sum())
        dice = 2 * intersection / (self.target_pixel_count + painted_count) if self.target_pixel_count + painted_count else 1.0
        if terminated and motion_failure is None:
            reward += 2.0 * dice
        sent_position = np.asarray(
            motion_info.pop("sent_xy_px", command_position), dtype=np.float32
        ).copy()
        info = {
            "character": self.character,
            "progress": self.step_index / len(self.trajectory),
            "dice": dice,
            "overflow_rate": (painted_count - intersection) / painted_count if painted_count else 0.0,
            "unpainted_rate": (self.target_pixel_count - intersection) / max(self.target_pixel_count, 1),
            "xy_offset_px": offset.copy(),
            "command_xy_px": command_position.copy(),
            "sent_xy_px": sent_position,
            "measured_xy_px": self.position.copy(),
            "post_measurement_valid": motion_failure is None,
            "tracking_error_px": (
                (self.position - sent_position).copy()
                if motion_failure is None else None
            ),
            "pressure": self.pressure,
            "measured_pressure": self.observation_pressure,
            "pressure_target": target_pressure,
            "pressure_delta": pressure_delta,
            "brush_width_px": brush_width,
            "contact": in_contact,
            "stroke_id": int(self.stroke_ids[self.step_index - 1]),
            "motion_failed": motion_failure is not None,
            "motion_failure": motion_failure,
        }
        info.update(motion_info)
        return self._observation(), reward, terminated, False, info

    def render(self):
        if self.canvas is None:
            return None
        return np.asarray(self.canvas.convert("RGB"))

    def _tangent(self, index):
        if not self.contact[index]:
            return np.zeros(2, dtype=np.float32)
        stroke_id = self.stroke_ids[index]
        if index + 1 < len(self.trajectory) and self.stroke_ids[index + 1] == stroke_id:
            vector = self.trajectory[index + 1, :2] - self.trajectory[index, :2]
        elif index > 0 and self.stroke_ids[index - 1] == stroke_id:
            vector = self.trajectory[index, :2] - self.trajectory[index - 1, :2]
        else:
            return np.zeros(2, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else np.zeros(2, dtype=np.float32)

    def _target_width(self, index, tangent=None):
        if not self.contact[index]:
            return 0.0
        width = self.target_cross_sections[index, 1]
        return float(width / (CANVAS_SIZE - 1))

    def _observation(self):
        index = min(self.step_index, len(self.trajectory) - 1)
        reference = self.trajectory[index, :2]
        actual = self.position
        tangent = self._tangent(index)
        target_width = self._target_width(index, tangent)
        return np.array(
            [
                actual[0] / (CANVAS_SIZE - 1),
                actual[1] / (CANVAS_SIZE - 1),
                reference[0] / (CANVAS_SIZE - 1),
                reference[1] / (CANVAS_SIZE - 1),
                (actual[0] - reference[0]) / self.deviation_scale,
                (actual[1] - reference[1]) / self.deviation_scale,
                self.observation_pressure,
                self.step_index / len(self.trajectory),
                self.stroke_progress[index],
                float(self.contact[index]),
                tangent[0],
                tangent[1],
                target_width,
            ],
            dtype=np.float32,
        )