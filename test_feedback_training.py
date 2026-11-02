import unittest
import tempfile
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal
from PIL import Image

from brush_env import (
    BrushCalligraphyEnv,
    OBSERVATION_CONVENTION,
    validate_observation_checkpoint,
    contact_pressure_to_z,
    pressure_from_width,
    pressure_to_z,
    width_from_pressure,
    z_to_contact_pressure,
    z_to_pressure,
)
from calligraphy_config import CANVAS_SIZE, ENDTIP_OFFSET_Z_M, PRESSURE_MAPPING_VERSION
from rl_train import collect_episode
from ros2_policy_runner import (
    apply_arm_z_offset,
    remove_arm_z_offset,
    validate_motion_checkpoint,
)
from ros2_feedback import (
    MotionUnreachableError,
    Ros2MotionBackend,
    base_to_pixel,
    cartesian_motion_duration,
    motion_wall_timeout,
    pixel_to_base,
)


class OffsetMotionBackend:
    def __init__(self, offset):
        self.offset = np.asarray(offset, dtype=np.float32)
        self.commands = []

    def reset(self, initial_position):
        return initial_position

    def step(self, command_position, pressure, contact):
        self.commands.append(command_position.copy())
        return command_position + self.offset


class UnreachableMotionBackend:
    def reset(self, initial_position):
        return initial_position

    def step(self, command_position, pressure, contact):
        raise MotionUnreachableError("test IK failure")


class MeasuredPressureBackend:
    def reset(self, initial_position):
        return {"measured_position": initial_position, "measured_pressure": 0.0}

    def step(self, command_position, pressure, contact):
        return {
            "measured_position": command_position,
            "info": {"measured_pressure": 0.25},
        }


class DiagnosticMotionBackend(OffsetMotionBackend):
    def step(self, command_position, pressure, contact):
        measured_position = super().step(command_position, pressure, contact)
        return {
            "measured_position": measured_position,
            "info": {"joint_state_sequence": 4},
        }


class ZeroPPOPolicy:
    def distribution(self, observations):
        mean = torch.zeros((observations.shape[0], 3), dtype=torch.float32)
        return Normal(mean, torch.ones_like(mean))

    def value(self, observations):
        return torch.zeros(observations.shape[0], dtype=torch.float32)


class FakeRunner:
    def __init__(self):
        self.args = SimpleNamespace(
            paper_center_x=0.19,
            paper_center_y=0.0,
            meters_per_pixel=0.0004,
            workspace_x_min=0.139,
            workspace_x_max=0.241,
            workspace_y_min=-0.051,
            workspace_y_max=0.051,
            command_hz=5.0,
            contact_z=0.0005,
            lift_z=0.0205,
            pressure_z_span_mm=15.0,
            z_transition_steps=1,
            arm_z_offset_mm=0.0,
        )
        self.command_pressure = 0.0
        self.target = None

    def joint_positions(self):
        return np.zeros(4)

    def fk(self, joints):
        return np.array([0.19, 0.0, 0.0205])

    def _command_z(self, contact_fraction=1.0):
        contact_z = contact_pressure_to_z(
            self.command_pressure, self.args.contact_z,
            self.args.pressure_z_span_mm / 1000.0,
        )
        return self.args.lift_z + contact_fraction * (contact_z - self.args.lift_z)

    def _measured_pressure(self, z_m, contact=True):
        if not contact:
            return 0.0
        return z_to_contact_pressure(
            z_m, self.args.contact_z, self.args.pressure_z_span_mm / 1000.0
        )

    def ik(self, target, seed):
        self.target = np.asarray(target)
        return np.ones(4)

    def send_joint_goal(self, joints, duration, timeout):
        return {
            "measured_joints": np.ones(4),
            "target_pose": self.target.copy(),
            "measured_pose": self.target + np.array([0.001, -0.002, 0.0]),
            "joint_state_sequence": 9,
            "joint_state_received_monotonic": 1.0,
            "joint_state_stamp_ns": 0,
            "simulation_rate": 0.2,
        }


class FeedbackEnvironmentTests(unittest.TestCase):
    def test_target_width_observation_preserves_width_independent_of_pressure(self):
        env = BrushCalligraphyEnv(characters=("ichi",))
        env.reset(options={"character": "ichi"})
        index = int(np.flatnonzero(env.contact)[0])
        env.step_index = index
        env.target_cross_sections[index, 1] = 40.0
        env.observation_pressure = 0.25
        observation = env._observation()
        self.assertAlmostEqual(observation[12], 40.0 / (CANVAS_SIZE - 1))
        self.assertAlmostEqual(observation[6], 0.25)
        env.contact[index] = False
        self.assertEqual(env._observation()[12], 0.0)

    def test_checkpoint_requires_current_observation_convention(self):
        with self.assertRaisesRegex(ValueError, "incompatible observation"):
            validate_observation_checkpoint({})
        validate_observation_checkpoint({"observation_convention": OBSERVATION_CONVENTION})

    def test_dataset_loader_ignores_csv_z_press(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "trajectory.csv"
            csv_path.write_text(
                "progress,x_px,y_px\n0,10,20\n1,30,40\n",
                encoding="utf-8",
            )
            image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
            image.putpixel((10, 20), 0)
            image.save(Path(directory) / "target.png")

            env = BrushCalligraphyEnv(artifact_dir=directory)
            dataset = env.datasets["custom"]

        self.assertEqual(dataset["trajectory"].shape[1], 2)
        np.testing.assert_allclose(
            dataset["trajectory"][[0, -1], :2], [[10, 20], [30, 40]]
        )

    def test_motion_checkpoint_pressure_mapping_version(self):
        args = SimpleNamespace(
            paper_center_x=0.19,
            paper_center_y=0.0,
            meters_per_pixel=0.0004,
            contact_z=0.0005,
            lift_z=0.0205,
            command_hz=5.0,
            max_speed_mm_s=40.0,
            pressure_z_span_mm=15.0,
            z_transition_steps=3,
            arm_z_offset_mm=0.0,
        )
        config = {
            "paper_center_base_m": [0.19, 0.0],
            "meters_per_pixel": 0.0004,
            "contact_z_m": 0.0005,
            "lift_z_m": 0.0205,
            "pressure_z_span_m": 0.015,
            "z_transition_steps": 3,
            "arm_z_offset_m": 0.0,
            "endtip_offset_z_m": ENDTIP_OFFSET_Z_M,
            "pressure_mapping_version": PRESSURE_MAPPING_VERSION,
            "command_hz": 5.0,
            "max_speed_mm_s": 40.0,
        }
        validate_motion_checkpoint(
            {"motion_backend": "gazebo", "motion_config": config}, args
        )
        config["endtip_offset_z_m"] = ENDTIP_OFFSET_Z_M + 0.002
        with self.assertRaisesRegex(ValueError, "endtip_offset_z_m"):
            validate_motion_checkpoint(
                {"motion_backend": "gazebo", "motion_config": config}, args
            )
        config["endtip_offset_z_m"] = ENDTIP_OFFSET_Z_M
        config["pressure_mapping_version"] = "contact_span_v1"
        with self.assertRaisesRegex(ValueError, "pressure_mapping_version"):
            validate_motion_checkpoint(
                {"motion_backend": "gazebo", "motion_config": config}, args
            )
        config["pressure_mapping_version"] = PRESSURE_MAPPING_VERSION
        config["pressure_z_span_m"] = 0.020
        with self.assertRaisesRegex(ValueError, "pressure_z_span_mm"):
            validate_motion_checkpoint(
                {"motion_backend": "gazebo", "motion_config": config}, args
            )

    def test_linear_z_pressure_and_width_mappings(self):
        contact_z, lift_z = 0.0005, 0.0205
        for pressure in (0.0, 0.1, 0.5, 1.0):
            z_value = pressure_to_z(pressure, contact_z, lift_z)
            self.assertAlmostEqual(z_to_pressure(z_value, contact_z, lift_z), pressure)

        self.assertAlmostEqual(pressure_to_z(0.0, contact_z, lift_z), lift_z)
        self.assertAlmostEqual(pressure_to_z(1.0, contact_z, lift_z), contact_z)
        self.assertEqual(width_from_pressure(0.0), 2)
        self.assertEqual(width_from_pressure(1.0), 20)
        self.assertAlmostEqual(pressure_from_width(6), 4.0 / 18.0)
        self.assertEqual(z_to_pressure(lift_z + 0.001, contact_z, lift_z), 0.0)
        self.assertEqual(z_to_pressure(contact_z - 0.001, contact_z, lift_z), 1.0)
        with self.assertRaises(ValueError):
            pressure_to_z(0.5, lift_z, contact_z)

        span = 0.015
        for pressure in (0.0, 0.1, 0.5, 1.0):
            z_value = contact_pressure_to_z(pressure, contact_z, span)
            self.assertAlmostEqual(
                z_to_contact_pressure(z_value, contact_z, span), pressure
            )
        self.assertAlmostEqual(contact_pressure_to_z(0.0, contact_z, span), contact_z)
        self.assertAlmostEqual(contact_pressure_to_z(0.5, contact_z, span), -0.007)
        self.assertAlmostEqual(contact_pressure_to_z(1.0, contact_z, span), -0.0145)
        self.assertEqual(z_to_contact_pressure(0.0205, contact_z, span), 0.0)
        self.assertEqual(z_to_contact_pressure(contact_z, contact_z, span), 0.0)
        self.assertEqual(z_to_contact_pressure(-0.0145, contact_z, span), 1.0)
        with self.assertRaises(ValueError):
            contact_pressure_to_z(0.5, contact_z, 0.0)

    def test_arm_z_offset_preserves_pressure_mapping(self):
        contact_z, lift_z = 0.0005, 0.0205
        offset_m = -0.010
        for pressure in (0.0, 0.1, 0.5, 1.0):
            mapped_z = pressure_to_z(pressure, contact_z, lift_z)
            commanded_z = apply_arm_z_offset(mapped_z, offset_m)
            corrected_z = remove_arm_z_offset(commanded_z, offset_m)
            self.assertAlmostEqual(commanded_z, mapped_z - 0.010)
            self.assertAlmostEqual(
                z_to_pressure(corrected_z, contact_z, lift_z), pressure
            )

    def test_measured_position_drives_observation_and_drawing(self):
        backend = OffsetMotionBackend([3.0, -4.0])
        env = BrushCalligraphyEnv(characters=("ichi",), motion_backend=backend)
        env.reset(options={"character": "ichi"})
        drawn_positions = []
        env._draw_brush_dab = lambda draw, position, *args: drawn_positions.append(
            np.asarray(position).copy()
        )

        observation, _, _, _, info = env.step(np.array([0.5, 0.0, 0.5]))

        expected_command = env.trajectory[0, :2] + np.array([6.0, 0.0])
        expected_measurement = expected_command + np.array([3.0, -4.0])
        np.testing.assert_allclose(backend.commands[0], expected_command)
        np.testing.assert_allclose(env.position, expected_measurement)
        np.testing.assert_allclose(drawn_positions[0], expected_measurement)
        np.testing.assert_allclose(info["command_xy_px"], expected_command)
        np.testing.assert_allclose(info["measured_xy_px"], expected_measurement)
        np.testing.assert_allclose(info["tracking_error_px"], [3.0, -4.0])
        np.testing.assert_allclose(
            observation[:2], expected_measurement / 255.0
        )
        self.assertEqual(observation.shape, (13,))
        self.assertIsNotNone(env._brush_heading)

        env.reset(options={"character": "ichi"})

        self.assertIsNone(env._brush_heading)

    def test_feedback_observation_space_allows_measured_position_outside_canvas(self):
        backend = OffsetMotionBackend([300.0, -200.0])
        env = BrushCalligraphyEnv(characters=("ichi",), motion_backend=backend)
        env.reset(options={"character": "ichi"})
        env._draw_brush_dab = lambda draw, position, *args: None

        observation, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))

        self.assertTrue(env.observation_space.contains(observation))
        self.assertGreater(observation[0], 1.0)
        self.assertLess(observation[1], 0.0)

    def test_ik_failure_terminates_without_painting(self):
        env = BrushCalligraphyEnv(
            characters=("ichi",), motion_backend=UnreachableMotionBackend()
        )
        env.reset(options={"character": "ichi"})

        _, reward, terminated, truncated, info = env.step(
            np.ones(3, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -1.0)
        self.assertTrue(info["motion_failed"])
        self.assertEqual(info["motion_failure"], "test IK failure")
        self.assertFalse(info["post_measurement_valid"])
        self.assertIsNone(info["tracking_error_px"])
        self.assertFalse((np.asarray(env.canvas) < 128).any())

    def test_observation_uses_measured_pressure_separately(self):
        env = BrushCalligraphyEnv(
            characters=("ichi",), motion_backend=MeasuredPressureBackend()
        )
        env.reset(options={"character": "ichi"})
        drawn_widths = []
        env._draw_brush_dab = lambda draw, position, width, *args: drawn_widths.append(width)

        observation, _, _, _, info = env.step(
            np.array([0.0, 0.0, 0.5], dtype=np.float32)
        )

        self.assertAlmostEqual(env.pressure, 0.5)
        self.assertAlmostEqual(observation[6], 0.25)
        self.assertAlmostEqual(info["measured_pressure"], 0.25)
        self.assertEqual(drawn_widths, [width_from_pressure(0.25)])

    def test_ros_motion_backend_returns_post_action_fk_position(self):
        x_m, y_m = pixel_to_base(31.0, 209.0, 0.19, 0.0, 0.0004)
        round_trip = base_to_pixel(x_m, y_m, 0.19, 0.0, 0.0004)
        np.testing.assert_allclose(round_trip, [31.0, 209.0], atol=1e-4)

        runner = FakeRunner()
        backend = Ros2MotionBackend(runner)
        result = backend.step([127.5, 127.5], 0.5, True)
        expected_measured = base_to_pixel(0.191, -0.002, 0.19, 0.0, 0.0004)

        np.testing.assert_allclose(result["measured_position"], expected_measured)
        self.assertGreater(result["info"]["tracking_error_m"], 0.0)
        self.assertEqual(result["info"]["joint_state_sequence"], 9)
        self.assertEqual(result["info"]["joint_state_stamp_ns"], 0)
        self.assertAlmostEqual(runner.target[2], -0.007)

    def test_ros_motion_backend_blends_contact_transitions(self):
        runner = FakeRunner()
        runner.args.z_transition_steps = 3
        backend = Ros2MotionBackend(runner)

        first_contact = backend.step([127.5, 127.5], 0.0, True)
        self.assertAlmostEqual(first_contact["info"]["contact_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(runner.target[2], 0.0205 + (0.0005 - 0.0205) / 3.0)

        backend.step([127.5, 127.5], 0.0, True)
        self.assertAlmostEqual(backend.contact_fraction, 2.0 / 3.0)

        first_lift = backend.step([127.5, 127.5], 0.0, False)
        self.assertAlmostEqual(first_lift["info"]["contact_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(runner.target[2], 0.0205 + (0.0005 - 0.0205) / 3.0)

    def test_motion_timeout_accounts_for_slow_simulation_clock(self):
        self.assertAlmostEqual(motion_wall_timeout(20.0, 0.2, 40.0), 110.0)
        self.assertAlmostEqual(motion_wall_timeout(1.0, 0.0, 10.0), 60.0)
        self.assertAlmostEqual(motion_wall_timeout(100.0, 0.01, 40.0), 1800.0)

    def test_reset_speed_reduces_initial_move_duration(self):
        distance_m = 0.34
        normal_duration = cartesian_motion_duration(distance_m, 15.0)
        faster_duration = cartesian_motion_duration(distance_m, 60.0)

        self.assertAlmostEqual(normal_duration / faster_duration, 4.0)
        self.assertEqual(cartesian_motion_duration(0.001, 60.0), 1.0)

    def test_ppo_collector_persists_motion_diagnostics(self):
        env = BrushCalligraphyEnv(
            characters=("ichi",), motion_backend=DiagnosticMotionBackend([2.0, -1.0])
        )
        result = collect_episode(
            env, ZeroPPOPolicy(), "ichi", gamma=0.99, gae_lambda=0.95, max_steps=1
        )

        self.assertEqual(len(result["motion_diagnostics"]), 1)
        diagnostic = result["motion_diagnostics"][0]
        self.assertEqual(diagnostic["step"], 0)
        self.assertTrue(diagnostic["post_measurement_valid"])
        self.assertEqual(diagnostic["tracking_error_px"], [2.0, -1.0])
        json.dumps(result["motion_diagnostics"])


if __name__ == "__main__":
    unittest.main()