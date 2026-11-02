#!/usr/bin/env python3
"""Run a baseline trajectory or fixed policy on a selected character with ROS feedback."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import JointConstraint
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from brush_env import (
    BrushCalligraphyEnv,
    contact_pressure_to_z,
    z_to_contact_pressure,
)
from calligraphy_config import (
    CANVAS_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_BASELINE_BRUSH_WIDTH_PX,
    ENDTIP_OFFSET_Z_M,
    HELD_OUT_CHARACTER,
    PRESSURE_MAPPING_VERSION,
    PROJECT_ROOT,
    TRAIN_CHARACTERS,
)
from evaluate_heldout import corrected_reference, load_bc_policy, load_rl_policy
from ros2_feedback import (
    MotionUnreachableError,
    Ros2MotionBackend,
    base_to_pixel,
    cartesian_motion_duration,
    motion_wall_timeout,
    pixel_to_base,
)


ARM_JOINTS = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
]
BASE_FRAME = "crane_plus_base"
TIP_LINK = "crane_plus_link_endtip"
GRIPPER_JOINT = "crane_plus_joint_hand"
GRIPPER_CLOSED_POSITION = 0.50


def apply_arm_z_offset(z_m, offset_m):
    return float(z_m) + float(offset_m)


def remove_arm_z_offset(z_m, offset_m):
    return float(z_m) - float(offset_m)


class GakuPolicyRunner(Node):
    def __init__(self, args, env, policy, policy_name, checkpoint):
        super().__init__("gaku_policy_runner")
        self.args = args
        self.env = env
        self.policy = policy
        self.policy_name = policy_name
        self.checkpoint = checkpoint
        self.motion_backend = Ros2MotionBackend(
            self, getattr(args, "max_speed_mm_s", 15.0)
        )
        reset_speed_mm_s = getattr(args, "reset_speed_mm_s", None)
        if reset_speed_mm_s is None:
            reset_speed_mm_s = 50.0
        if reset_speed_mm_s <= 0.0:
            raise ValueError("reset speed must be positive")
        self.reset_speed_mm_s = float(reset_speed_mm_s)
        self.arm_z_offset_m = float(args.arm_z_offset_mm) / 1000.0
        self.latest_joint_state = None
        self.latest_joint_state_time = None
        self.latest_joint_state_stamp_ns = None
        self.simulation_rate = 1.0
        self.has_simulation_rate_sample = False
        self.joint_state_sequence = 0
        self.command_pressure = 0.0
        self.rows = []
        self.error = None
        self.texture_publisher = (
            self.create_publisher(RosImage, args.texture_topic, 1)
            if args.texture_topic else None
        )

        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.trajectory_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/crane_plus_arm_controller/follow_joint_trajectory",
        )
        self.gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/crane_plus_gripper_controller/follow_joint_trajectory",
        )
        self.fk_client = self.create_client(GetPositionFK, "/compute_fk")
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        for name, client in (("/compute_fk", self.fk_client), ("/compute_ik", self.ik_client)):
            if not client.wait_for_service(timeout_sec=20.0):
                raise RuntimeError(f"ROS service {name} is unavailable")
        if not self.trajectory_action.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("FollowJointTrajectory action is unavailable")
        if getattr(args, "require_gripper_controller", True) and not self.gripper_action.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("Gripper FollowJointTrajectory action is unavailable")
        self._warm_up_joint_state_rate()

    def _warm_up_joint_state_rate(self, minimum_samples=10, timeout=10.0):
        deadline = time.monotonic() + timeout
        while self.joint_state_sequence < minimum_samples:
            if time.monotonic() >= deadline:
                raise TimeoutError("Could not warm up /joint_states simulation clock")
            rclpy.spin_once(self, timeout_sec=0.1)

    def _on_joint_state(self, message):
        received_at = time.monotonic()
        stamp_ns = self._joint_state_stamp_ns(message)
        if (
            self.latest_joint_state_stamp_ns is not None
            and stamp_ns > self.latest_joint_state_stamp_ns
        ):
            wall_delta = received_at - self.latest_joint_state_time
            if wall_delta > 0.0:
                observed_rate = (stamp_ns - self.latest_joint_state_stamp_ns) / (
                    1e9 * wall_delta
                )
                if 0.01 <= observed_rate <= 2.0:
                    if self.has_simulation_rate_sample:
                        self.simulation_rate = (
                            0.8 * self.simulation_rate + 0.2 * observed_rate
                        )
                    else:
                        self.simulation_rate = observed_rate
                        self.has_simulation_rate_sample = True
        self.latest_joint_state = message
        self.latest_joint_state_time = received_at
        self.latest_joint_state_stamp_ns = stamp_ns
        self.joint_state_sequence += 1

    @staticmethod
    def _joint_state_stamp_ns(message):
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def joint_positions(self, timeout=5.0, after_sequence=None, after_stamp_ns=None):
        deadline = time.monotonic() + timeout
        while (
            self.latest_joint_state is None
            or time.monotonic() - self.latest_joint_state_time > 1.0
            or (
                after_sequence is not None
                and self.joint_state_sequence <= after_sequence
            )
            or (
                after_stamp_ns is not None
                and self._joint_state_stamp_ns(self.latest_joint_state) > 0
                and self._joint_state_stamp_ns(self.latest_joint_state) <= after_stamp_ns
            )
        ):
            if time.monotonic() >= deadline:
                raise TimeoutError("No new /joint_states message after motion")
            rclpy.spin_once(self, timeout_sec=0.05)
        by_name = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        missing = [name for name in ARM_JOINTS if name not in by_name]
        if missing:
            raise RuntimeError(f"/joint_states is missing arm joints: {missing}")
        return np.asarray([by_name[name] for name in ARM_JOINTS], dtype=np.float64)

    def _call_service(self, client, request, name):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            raise TimeoutError(f"ROS service {name} timed out")
        response = future.result()
        if response is None:
            raise RuntimeError(f"ROS service {name} returned no response")
        return response

    def fk(self, joints):
        request = GetPositionFK.Request()
        request.header.frame_id = BASE_FRAME
        request.fk_link_names = [TIP_LINK]
        request.robot_state.joint_state.name = ARM_JOINTS
        request.robot_state.joint_state.position = [float(value) for value in joints]
        request.robot_state.is_diff = False
        response = self._call_service(self.fk_client, request, "/compute_fk")
        if response.error_code.val != 1 or not response.pose_stamped:
            raise RuntimeError(f"MoveIt FK failed with code {response.error_code.val}")
        pose = response.pose_stamped[0].pose
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
        )

    def ik(self, target_position, seed, joint_tolerance=None):
        request = GetPositionIK.Request()
        ik_request = request.ik_request
        ik_request.group_name = "arm"
        ik_request.ik_link_name = TIP_LINK
        ik_request.pose_stamped.header.frame_id = BASE_FRAME
        ik_request.pose_stamped.pose.position.x = float(target_position[0])
        ik_request.pose_stamped.pose.position.y = float(target_position[1])
        ik_request.pose_stamped.pose.position.z = float(target_position[2])
        ik_request.pose_stamped.pose.orientation.w = 1.0
        ik_request.robot_state.joint_state.name = ARM_JOINTS
        ik_request.robot_state.joint_state.position = [float(value) for value in seed]
        ik_request.robot_state.is_diff = False
        ik_request.avoid_collisions = False
        ik_request.timeout = Duration(sec=1, nanosec=0)
        if joint_tolerance is not None:
            for name, position in zip(ARM_JOINTS, seed):
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = float(position)
                constraint.tolerance_above = joint_tolerance
                constraint.tolerance_below = joint_tolerance
                constraint.weight = 1.0
                ik_request.constraints.joint_constraints.append(constraint)
        response = self._call_service(self.ik_client, request, "/compute_ik")
        if response.error_code.val != 1:
            raise MotionUnreachableError(
                f"IK failed at {list(target_position)} with code {response.error_code.val}"
            )
        solution = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        return np.asarray([solution[name] for name in ARM_JOINTS], dtype=np.float64)

    def send_joint_goal(self, joints, duration, timeout):
        wall_timeout = motion_wall_timeout(
            duration, self.simulation_rate, timeout
        )
        command_start_stamp_ns = (
            self._joint_state_stamp_ns(self.latest_joint_state)
            if self.latest_joint_state is not None else 0
        )
        goal = FollowJointTrajectory.Goal()
        message = goal.trajectory
        message.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in joints]
        point.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1.0) * 1e9)
        )
        message.points = [point]
        goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
        accepted_future = self.trajectory_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self, accepted_future, timeout_sec=wall_timeout
        )
        if not accepted_future.done():
            raise TimeoutError("FollowJointTrajectory goal acceptance timed out")
        handle = accepted_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("FollowJointTrajectory goal was rejected")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=wall_timeout
        )
        if not result_future.done():
            handle.cancel_goal_async()
            raise TimeoutError("FollowJointTrajectory result timed out")
        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
            raise RuntimeError(
                f"FollowJointTrajectory failed: status={result.status}, "
                f"error={result.result.error_code} {result.result.error_string}"
            )
        target_pose = self.fk(joints)
        after_sequence = self.joint_state_sequence
        after_stamp_ns = command_start_stamp_ns
        deadline = time.monotonic() + 1.5
        position_error = float("inf")
        while time.monotonic() < deadline:
            try:
                current = self.joint_positions(
                    timeout=min(0.2, max(0.01, deadline - time.monotonic())),
                    after_sequence=after_sequence,
                    after_stamp_ns=after_stamp_ns,
                )
            except TimeoutError:
                continue
            after_sequence = self.joint_state_sequence
            measured_pose = self.fk(current)
            position_error = float(np.linalg.norm(measured_pose - target_pose))
            if position_error < 0.005:
                return {
                    "measured_joints": current.copy(),
                    "measured_pose": measured_pose.copy(),
                    "target_pose": target_pose.copy(),
                    "position_error_m": position_error,
                    "joint_state_sequence": self.joint_state_sequence,
                    "joint_state_received_monotonic": self.latest_joint_state_time,
                    "joint_state_stamp_ns": self._joint_state_stamp_ns(
                        self.latest_joint_state
                    ),
                    "simulation_rate": self.simulation_rate,
                }
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            f"Action succeeded but measured tip missed the goal by "
            f"{position_error * 1000.0:.1f} mm"
        )

    def close_gripper(self):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [GRIPPER_JOINT]
        point = JointTrajectoryPoint()
        point.positions = [GRIPPER_CLOSED_POSITION]
        point.time_from_start = Duration(sec=1, nanosec=0)
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
        accepted_future = self.gripper_action.send_goal_async(goal)
        handle = None
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)
            if accepted_future.done():
                handle = accepted_future.result()
                break
        if handle is None:
            self.get_logger().warn("Gripper close goal was not accepted before replay continued")
            return
        if handle is None or not handle.accepted:
            raise RuntimeError("Gripper close goal was rejected")
        result_future = handle.get_result_async()
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)
            if result_future.done():
                break
        if not result_future.done():
            self.get_logger().warn("Gripper close result did not complete; continuing without timeout")
            return
        result = result_future.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code != 0:
            raise RuntimeError(
                f"Gripper close failed: status={result.status}, "
                f"error={result.result.error_code} {result.result.error_string}"
            )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = self.latest_joint_state
            if state is not None:
                positions = dict(zip(state.name, state.position))
                if GRIPPER_JOINT in positions and abs(
                    positions[GRIPPER_JOINT] - GRIPPER_CLOSED_POSITION
                ) < 0.03:
                    return
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warn("Measured gripper joint missed the close position; continuing")

    def move_to_first_lift(self):
        reference = self.env.trajectory[0, :2]
        x_m, y_m = pixel_to_base(
            reference[0], reference[1],
            self.args.paper_center_x,
            self.args.paper_center_y,
            self.args.meters_per_pixel,
        )
        current = self.joint_positions()
        current_pose = self.fk(current)
        target_pose = np.array(
            [x_m, y_m, apply_arm_z_offset(self.args.lift_z, self.arm_z_offset_m)],
            dtype=np.float64,
        )
        solution = self.ik(target_pose, current)
        duration = cartesian_motion_duration(
            float(np.linalg.norm(target_pose - current_pose)),
            self.reset_speed_mm_s,
        )
        self.send_joint_goal(solution, duration=duration, timeout=40.0)

    def _measured_pressure(self, z_m, contact=True):
        if not contact:
            return 0.0
        corrected_z = remove_arm_z_offset(z_m, self.arm_z_offset_m)
        return z_to_contact_pressure(
            corrected_z,
            self.args.contact_z,
            self.args.pressure_z_span_mm / 1000.0,
        )

    def _command_z(self, contact_fraction=1.0):
        contact_z = contact_pressure_to_z(
            self.command_pressure,
            self.args.contact_z,
            self.args.pressure_z_span_mm / 1000.0,
        )
        contact_fraction = float(np.clip(contact_fraction, 0.0, 1.0))
        target_z = self.args.lift_z + contact_fraction * (
            contact_z - self.args.lift_z
        )
        return apply_arm_z_offset(
            target_z, self.arm_z_offset_m
        )

    def _raise_tip(self, current_pose, current_joints):
        self.command_pressure = 0.0
        target = [
            current_pose[0],
            current_pose[1],
            apply_arm_z_offset(self.args.lift_z, self.arm_z_offset_m),
        ]
        lift_joints = self.ik(target, current_joints)
        duration = max(
            1.0,
            float(np.linalg.norm(np.asarray(target) - current_pose))
            / self.motion_backend.max_speed_m_s,
        )
        self.send_joint_goal(lift_joints, duration=duration, timeout=15.0)
        measured_pose = self.fk(self.joint_positions())
        corrected_z = remove_arm_z_offset(measured_pose[2], self.arm_z_offset_m)
        if corrected_z < self.args.lift_z - 0.005:
            raise RuntimeError(
                f"Pen lift did not reach clearance: z={measured_pose[2]:.4f} m"
            )

    def _observation(self, step_index, actual_pixel, pressure):
        self.env.step_index = step_index
        self.env.position = actual_pixel.astype(np.float32)
        self.env.pressure = pressure
        self.env.observation_pressure = pressure
        return self.env._observation()

    def _policy_action(self, observation):
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.policy_name == "bc":
                action = self.policy(tensor).squeeze(0).numpy()
            else:
                action = self.policy.deterministic_action(tensor).squeeze(0).numpy()
                action[2] = (action[2] + 1.0) / 2.0
        return np.asarray(action, dtype=np.float32)

    def _draw_measurement(self, step_index, actual_pixel, render_pressure):
        self.env.position = actual_pixel.astype(np.float32)
        self.env.pressure = render_pressure
        previous = self.env._previous_position.copy()
        if self.env.contact[step_index]:
            same_stroke_before = (
                step_index > 0
                and self.env.contact[step_index - 1]
                and self.env.stroke_ids[step_index - 1] == self.env.stroke_ids[step_index]
            )
            tangent = self.env._brush_tangent(
                self.env._brush_reference_tangent(step_index), same_stroke_before
            )
            width = self.env.brush_width_for_pressure(render_pressure)
            speed = float(np.linalg.norm(actual_pixel - previous))
            draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(self.env.canvas)
            self.env._draw_brush_dab(
                draw, actual_pixel, width, tangent, speed
            )
        self.env._previous_position = actual_pixel.astype(np.float32)
        if self.texture_publisher is not None:
            pixels = np.asarray(self.env.canvas.convert("RGB"), dtype=np.uint8)
            message = RosImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "paper"
            message.height, message.width = pixels.shape[:2]
            message.encoding = "rgb8"
            message.step = message.width * 3
            message.data = pixels.tobytes()
            self.texture_publisher.publish(message)

    def execute(self):
        self.env.reset(options={"character": self.args.character})
        if getattr(self.args, "close_gripper", True):
            self.close_gripper()
        self.move_to_first_lift()
        initial_joints = self.joint_positions()
        initial_pose = self.fk(initial_joints)
        self.env._previous_position = base_to_pixel(
            initial_pose[0], initial_pose[1], self.args.paper_center_x,
            self.args.paper_center_y, self.args.meters_per_pixel,
        )

        total_steps = len(self.env.trajectory)
        run_steps = min(total_steps, self.args.max_steps or total_steps)
        max_pressure_delta = (
            float(self.checkpoint["max_pressure_rate_per_second"])
            / self.args.command_hz
        )
        self.get_logger().info(
            f"Running {self.policy_name} on {self.args.character}: "
            f"{run_steps}/{total_steps} feedback steps"
        )
        baseline_pressure = self.env.pressure_for_brush_width(
            DEFAULT_BASELINE_BRUSH_WIDTH_PX
        )

        for step_index in range(run_steps):
            rclpy.spin_once(self, timeout_sec=0.01)
            joints = self.joint_positions()
            actual_pose = self.fk(joints)
            actual_pixel = base_to_pixel(
                actual_pose[0], actual_pose[1],
                self.args.paper_center_x,
                self.args.paper_center_y,
                self.args.meters_per_pixel,
            )
            is_contact = bool(self.env.contact[step_index])
            pressure = self._measured_pressure(actual_pose[2], is_contact)
            reference = self.env.trajectory[step_index, :2]
            if self.policy_name == "baseline":
                target_pixel = corrected_reference(self.env, step_index)
                target_pressure = baseline_pressure if is_contact else 0.0
            else:
                observation = self._observation(step_index, actual_pixel, pressure)
                action = self._policy_action(observation)
                if not is_contact:
                    action = np.zeros(3, dtype=np.float32)
                target_pixel = reference + action[:2] * self.env.max_xy_offset_px
                target_pressure = float(action[2])

            pressure_delta = np.clip(
                target_pressure - self.command_pressure,
                -max_pressure_delta,
                max_pressure_delta,
            )
            self.command_pressure = float(
                np.clip(self.command_pressure + pressure_delta, 0.0, 1.0)
            )
            pre_actual_pixel = actual_pixel.copy()
            pre_actual_pose = actual_pose.copy()
            pre_joints = joints.copy()
            motion_result = self.motion_backend.step(
                target_pixel, self.command_pressure, is_contact
            )
            actual_pixel = motion_result["measured_position"]
            motion_info = motion_result["info"]
            actual_pose = np.asarray(motion_info["measured_xyz_m"], dtype=np.float64)
            measured_joints = np.asarray(
                motion_info["measured_joints_rad"], dtype=np.float64
            )
            target_x, target_y, target_z = motion_info["requested_xyz_m"]
            pressure = self._measured_pressure(actual_pose[2], is_contact)
            self._draw_measurement(step_index, actual_pixel, pressure)
            display_reference = corrected_reference(self.env, step_index)
            row = {
                "step": step_index,
                "stroke_id": int(self.env.stroke_ids[step_index]),
                "contact": int(is_contact),
                "reference_x_px": float(display_reference[0]),
                "reference_y_px": float(display_reference[1]),
                "policy_reference_x_px": float(reference[0]),
                "policy_reference_y_px": float(reference[1]),
                "command_x_px": float(target_pixel[0]),
                "command_y_px": float(target_pixel[1]),
                "sent_x_px": float(motion_info["sent_xy_px"][0]),
                "sent_y_px": float(motion_info["sent_xy_px"][1]),
                "pre_actual_x_px": float(pre_actual_pixel[0]),
                "pre_actual_y_px": float(pre_actual_pixel[1]),
                "pre_actual_x_base_m": float(pre_actual_pose[0]),
                "pre_actual_y_base_m": float(pre_actual_pose[1]),
                "pre_actual_z_base_m": float(pre_actual_pose[2]),
                "actual_x_px": float(actual_pixel[0]),
                "actual_y_px": float(actual_pixel[1]),
                "actual_x_base_m": float(actual_pose[0]),
                "actual_y_base_m": float(actual_pose[1]),
                "actual_z_base_m": float(actual_pose[2]),
                "pressure_estimate": pressure,
                "command_pressure": self.command_pressure,
                "target_x_base_m": target_x,
                "target_y_base_m": target_y,
                "target_z_base_m": target_z,
                "contact_fraction": motion_info["contact_fraction"],
                "ik_residual_m": motion_info["ik_residual_m"],
                "tracking_error_m": motion_info["tracking_error_m"],
                "command_duration_s": motion_info["command_duration_s"],
                "joint_state_sequence": motion_info["joint_state_sequence"],
                "joint_state_stamp_ns": motion_info["joint_state_stamp_ns"],
                "joint_state_received_monotonic": motion_info[
                    "joint_state_received_monotonic"
                ],
                "simulation_rate": motion_info["simulation_rate"],
            }
            row.update({
                f"pre_{name}_rad": float(value)
                for name, value in zip(ARM_JOINTS, pre_joints)
            })
            row.update({
                f"{name}_rad": float(value)
                for name, value in zip(ARM_JOINTS, measured_joints)
            })
            self.rows.append(row)
            if not is_contact:
                measured_z = remove_arm_z_offset(
                    actual_pose[2], self.arm_z_offset_m
                )
                if (
                    motion_info["contact_fraction"] == 0.0
                    and measured_z < self.args.lift_z - 0.005
                ):
                    raise RuntimeError(
                        f"Lift clearance lost during travel: z={measured_z:.4f} m"
                    )

            if step_index % 20 == 0:
                self.get_logger().info(
                    f"step {step_index}/{run_steps}, "
                    f"contact={int(is_contact)}, measured_xy_px="
                    f"({actual_pixel[0]:.1f}, {actual_pixel[1]:.1f})"
                )

        current = self.joint_positions()
        final_pose = self.fk(current)
        self._raise_tip(final_pose, current)

    def save_outputs(self):
        output_dir = self.args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        character = self.args.character
        measured_metrics = None
        if self.env.canvas is not None:
            painted = np.asarray(self.env.canvas) < 128
            intersection = int((painted & self.env.target_mask).sum())
            painted_count = int(painted.sum())
            target_count = max(self.env.target_pixel_count, 1)
            measured_metrics = {
                "dice": 2.0 * intersection / (self.env.target_pixel_count + painted_count)
                if self.env.target_pixel_count + painted_count else 1.0,
                "overflow_rate": (painted_count - intersection) / painted_count
                if painted_count else 0.0,
                "unpainted_rate": (self.env.target_pixel_count - intersection) / target_count,
            }
        csv_path = output_dir / f"{character}_measured_trajectory.csv"
        fieldnames = list(self.rows[0]) if self.rows else ["step"]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        if self.env.canvas is not None:
            self.env.render()
            self.env.target.save(output_dir / f"{character}_target.png")
            self.env.canvas.save(output_dir / f"{character}_measured.png")
        metadata = {
            "character": character,
            "policy": self.policy_name,
            "trajectory_source": "center_corrected_reference" if self.policy_name == "baseline" else "policy",
            "bc_checkpoint": str(self.args.bc_checkpoint),
            "rl_checkpoint": str(self.args.rl_checkpoint),
            "control_hz": self.checkpoint["control_hz"],
            "command_hz": self.args.command_hz,
            "meters_per_pixel": self.args.meters_per_pixel,
            "paper_center_base_m": [self.args.paper_center_x, self.args.paper_center_y],
            "contact_z_base_m": self.args.contact_z,
            "lift_z_base_m": self.args.lift_z,
            "pressure_z_span_m": self.args.pressure_z_span_mm / 1000.0,
            "z_transition_steps": self.args.z_transition_steps,
            "arm_z_offset_m": self.arm_z_offset_m,
            "pressure_mapping_version": PRESSURE_MAPPING_VERSION,
            "max_speed_mm_s": self.args.max_speed_mm_s,
            "feedback_source": "/joint_states -> MoveIt FK",
            "base_frame": BASE_FRAME,
            "tip_link": TIP_LINK,
            "endtip_offset_z_m": ENDTIP_OFFSET_Z_M,
            "measured_canvas_metrics": measured_metrics,
            "coordinate_mapping_calibrated": False,
            "physical_brush_model": False,
            "steps_saved": len(self.rows),
            "error": self.error,
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("baseline", "bc", "bc_rl"), default="bc_rl")
    parser.add_argument(
        "--character", choices=(*TRAIN_CHARACTERS, HELD_OUT_CHARACTER),
        default=HELD_OUT_CHARACTER,
    )
    parser.add_argument(
        "--bc-checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts/day10_endpoint_weight_bc/bc_policy.pt",
    )
    parser.add_argument(
        "--rl-checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts/day10_endpoint_weight_rl/bc_rl_policy.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts/day9_ros_gaku_position_only",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--texture-topic", default="/paper/texture_update",
        help="publish the measured canvas as rgb8 for a Gazebo texture uploader",
    )
    parser.add_argument(
        "--close-gripper", dest="close_gripper", action="store_true", default=True,
        help="close the gripper before replay starts (default: enabled)",
    )
    parser.add_argument(
        "--no-close-gripper", dest="close_gripper", action="store_false",
        help="skip the gripper close before replay starts",
    )
    parser.add_argument("--command-hz", type=float, default=5.0)
    parser.add_argument("--max-speed-mm-s", type=float, default=15.0)
    parser.add_argument(
        "--reset-speed-mm-s", type=float, default=50.0,
        help="speed for the initial move to the first lifted task position (default: 50 mm/s)",
    )
    parser.add_argument("--paper-center-x", type=float, default=0.10)
    parser.add_argument("--paper-center-y", type=float, default=0.0)
    parser.add_argument("--meters-per-pixel", type=float, default=0.0004)
    parser.add_argument(
        "--arm-z-offset-mm", type=float, default=-10.0,
        help="offset arm Z targets in millimeters; negative lowers the arm (default: -10)",
    )
    parser.add_argument(
        "--paper-surface-z", "--contact-z", dest="contact_z", type=float,
        default=0.0005,
        help="paper top in the robot base frame (default matches the SDF geometry)",
    )
    parser.add_argument("--lift-z", type=float, default=0.0205)
    parser.add_argument("--pressure-z-span-mm", type=float, default=15.0)
    parser.add_argument("--z-transition-steps", type=int, default=3)
    parser.add_argument("--workspace-x-min", type=float)
    parser.add_argument("--workspace-x-max", type=float)
    parser.add_argument("--workspace-y-min", type=float)
    parser.add_argument("--workspace-y-max", type=float)
    return parser.parse_args()


def validate_motion_checkpoint(checkpoint, args):
    if checkpoint.get("motion_backend") != "gazebo":
        return
    config = checkpoint.get("motion_config")
    if not isinstance(config, dict):
        raise ValueError("Gazebo-trained checkpoint is missing motion_config")
    expected = {
        "paper_center_x": args.paper_center_x,
        "paper_center_y": args.paper_center_y,
        "meters_per_pixel": args.meters_per_pixel,
        "contact_z": args.contact_z,
        "lift_z": args.lift_z,
        "pressure_z_span_mm": args.pressure_z_span_mm,
        "z_transition_steps": args.z_transition_steps,
        "arm_z_offset_mm": args.arm_z_offset_mm,
        "pressure_mapping_version": PRESSURE_MAPPING_VERSION,
        "endtip_offset_z_m": ENDTIP_OFFSET_Z_M,
        "command_hz": args.command_hz,
        "max_speed_mm_s": args.max_speed_mm_s,
    }
    recorded = {
        "paper_center_x": config.get("paper_center_base_m", [None, None])[0],
        "paper_center_y": config.get("paper_center_base_m", [None, None])[1],
        "meters_per_pixel": config.get("meters_per_pixel"),
        "contact_z": config.get("contact_z_m"),
        "lift_z": config.get("lift_z_m"),
        "pressure_z_span_mm": 1000.0 * config.get("pressure_z_span_m", float("nan")),
        "z_transition_steps": config.get("z_transition_steps"),
        "arm_z_offset_mm": 1000.0 * config.get("arm_z_offset_m", float("nan")),
        "pressure_mapping_version": config.get("pressure_mapping_version"),
        "endtip_offset_z_m": config.get("endtip_offset_z_m"),
        "command_hz": config.get("command_hz"),
        "max_speed_mm_s": config.get("max_speed_mm_s"),
    }
    mismatches = [
        name for name, value in expected.items()
        if (
            recorded[name] != value
            if name == "pressure_mapping_version"
            else recorded[name] is None or not np.isclose(value, recorded[name])
        )
    ]
    if mismatches:
        raise ValueError(
            "Gazebo checkpoint motion settings differ from runner arguments: "
            + ", ".join(mismatches)
        )


def main():
    args = parse_args()
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.command_hz <= 0.0:
        raise ValueError("--command-hz must be positive")
    if args.max_speed_mm_s <= 0.0:
        raise ValueError("--max-speed-mm-s must be positive")
    if not np.isfinite(args.arm_z_offset_mm):
        raise ValueError("--arm-z-offset-mm must be finite")
    if not np.isfinite(args.pressure_z_span_mm) or args.pressure_z_span_mm <= 0.0:
        raise ValueError("--pressure-z-span-mm must be positive and finite")
    if args.z_transition_steps < 1:
        raise ValueError("--z-transition-steps must be positive")
    if args.reset_speed_mm_s is not None and args.reset_speed_mm_s <= 0.0:
        raise ValueError("--reset-speed-mm-s must be positive")
    if args.lift_z <= args.contact_z:
        raise ValueError("--lift-z must be above --paper-surface-z")
    paper_half_extent = (CANVAS_SIZE - 1) / 2.0 * args.meters_per_pixel
    if args.workspace_x_min is None:
        args.workspace_x_min = args.paper_center_x - paper_half_extent
    if args.workspace_x_max is None:
        args.workspace_x_max = args.paper_center_x + paper_half_extent
    if args.workspace_y_min is None:
        args.workspace_y_min = args.paper_center_y - paper_half_extent
    if args.workspace_y_max is None:
        args.workspace_y_max = args.paper_center_y + paper_half_extent
    torch.set_num_threads(1)

    bc_checkpoint, bc_policy = load_bc_policy(args.bc_checkpoint)
    if args.policy == "bc":
        checkpoint, policy = bc_checkpoint, bc_policy
    elif args.policy == "bc_rl":
        checkpoint, policy = load_rl_policy(args.rl_checkpoint, bc_checkpoint)
    else:
        checkpoint, policy = bc_checkpoint, None
    validate_motion_checkpoint(checkpoint, args)

    env = BrushCalligraphyEnv(
        artifact_dir=args.data_dir,
        characters=(args.character,),
        max_xy_offset_px=bc_checkpoint["max_xy_offset_px"],
        control_hz=bc_checkpoint["control_hz"],
        max_pressure_rate_per_second=bc_checkpoint["max_pressure_rate_per_second"],
        lock_full_pressure=bc_checkpoint.get("lock_full_pressure", 0.5),
        endpoint_weight=bc_checkpoint.get("endpoint_weight", 1.0),
        path_weight=bc_checkpoint.get("path_weight", 1.0),
        render_mode="rgb_array",
    )
    rclpy.init()
    node = None
    try:
        node = GakuPolicyRunner(args, env, policy, args.policy, checkpoint)
        try:
            node.execute()
        except Exception as error:
            node.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            node.save_outputs()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()