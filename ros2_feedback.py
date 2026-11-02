"""ROS-backed motion feedback for calligraphy training and replay."""

import numpy as np

from calligraphy_config import CANVAS_SIZE


class MotionUnreachableError(RuntimeError):
    """The requested pose has no valid IK solution."""


def motion_wall_timeout(duration_s, simulation_rate, minimum_timeout_s, maximum_timeout_s=1800.0):
    estimated_wall_time = (float(duration_s) + 2.0) / max(
        float(simulation_rate), 0.05
    )
    return min(
        float(maximum_timeout_s),
        max(float(minimum_timeout_s), estimated_wall_time),
    )


def cartesian_motion_duration(distance_m, speed_mm_s, minimum_duration_s=1.0):
    if distance_m < 0.0 or speed_mm_s <= 0.0 or minimum_duration_s < 0.0:
        raise ValueError("distance and minimum duration must be non-negative; speed must be positive")
    return max(float(minimum_duration_s), float(distance_m) / (float(speed_mm_s) / 1000.0))


def pixel_to_base(x_px, y_px, center_x, center_y, meters_per_pixel):
    center_px = (CANVAS_SIZE - 1) / 2.0
    return (
        center_x - (y_px - center_px) * meters_per_pixel,
        center_y - (x_px - center_px) * meters_per_pixel,
    )


def base_to_pixel(x_m, y_m, center_x, center_y, meters_per_pixel):
    center_px = (CANVAS_SIZE - 1) / 2.0
    return np.array(
        [
            center_px - (y_m - center_y) / meters_per_pixel,
            center_px - (x_m - center_x) / meters_per_pixel,
        ],
        dtype=np.float32,
    )


class Ros2MotionBackend:
    """Adapt a ROS policy runner to BrushCalligraphyEnv's motion interface."""

    def __init__(self, runner, max_speed_mm_s=15.0):
        if max_speed_mm_s <= 0.0:
            raise ValueError("max_speed_mm_s must be positive")
        self.runner = runner
        self.max_speed_m_s = float(max_speed_mm_s) / 1000.0
        self.z_transition_steps = int(
            getattr(runner.args, "z_transition_steps", 1)
        )
        if self.z_transition_steps < 1:
            raise ValueError("z_transition_steps must be positive")
        self.contact_fraction = 0.0

    def _measured_position(self, pose):
        args = self.runner.args
        return base_to_pixel(
            pose[0], pose[1], args.paper_center_x, args.paper_center_y,
            args.meters_per_pixel,
        )

    def reset(self, initial_position):
        self.runner.command_pressure = 0.0
        self.contact_fraction = 0.0
        joints = self.runner.joint_positions()
        pose = self.runner.fk(joints)
        if pose[2] < self.runner.args.lift_z - 0.005:
            self.runner._raise_tip(pose, joints)
        self.runner.move_to_first_lift()
        joints = self.runner.joint_positions()
        pose = self.runner.fk(joints)
        return {
            "measured_position": self._measured_position(pose),
            "measured_pressure": self.runner._measured_pressure(
                pose[2], contact=False
            ),
        }

    def step(self, command_position, pressure, contact):
        runner = self.runner
        args = runner.args
        command_position = np.asarray(command_position, dtype=np.float32)
        sent_position = np.clip(command_position, 0.0, CANVAS_SIZE - 1.0)
        target_x, target_y = pixel_to_base(
            sent_position[0], sent_position[1], args.paper_center_x,
            args.paper_center_y, args.meters_per_pixel,
        )
        target_x = float(np.clip(target_x, args.workspace_x_min, args.workspace_x_max))
        target_y = float(np.clip(target_y, args.workspace_y_min, args.workspace_y_max))
        sent_position = base_to_pixel(
            target_x, target_y, args.paper_center_x, args.paper_center_y,
            args.meters_per_pixel,
        )

        current_joints = runner.joint_positions()
        current_pose = runner.fk(current_joints)
        runner.command_pressure = float(np.clip(pressure, 0.0, 1.0))
        target_fraction = 1.0 if contact else 0.0
        fraction_step = 1.0 / self.z_transition_steps
        if self.contact_fraction < target_fraction:
            self.contact_fraction = min(
                target_fraction, self.contact_fraction + fraction_step
            )
        elif self.contact_fraction > target_fraction:
            self.contact_fraction = max(
                target_fraction, self.contact_fraction - fraction_step
            )
        target_z = runner._command_z(self.contact_fraction)
        target_xyz = np.array([target_x, target_y, target_z], dtype=np.float64)
        target_joints = runner.ik(target_xyz, current_joints)
        duration = max(
            1.0 / args.command_hz,
            float(np.linalg.norm(target_xyz - current_pose)) / self.max_speed_m_s,
        )
        result = runner.send_joint_goal(
            target_joints,
            duration=duration,
            timeout=max(10.0, duration + 2.0),
        )
        if result is None:
            raise RuntimeError("ROS runner did not return a post-command measurement")

        measured_pose = result["measured_pose"]
        measured_position = self._measured_position(measured_pose)
        ik_pose = result["target_pose"]
        return {
            "measured_position": measured_position,
            "info": {
                "sent_xy_px": sent_position.copy(),
                "requested_xyz_m": [float(target_x), float(target_y), float(target_z)],
                "contact_fraction": float(self.contact_fraction),
                "ik_fk_xyz_m": [float(value) for value in ik_pose],
                "measured_xyz_m": [float(value) for value in measured_pose],
                "measured_pressure": float(
                    runner._measured_pressure(measured_pose[2], contact=contact)
                ),
                "measured_joints_rad": [
                    float(value) for value in result["measured_joints"]
                ],
                "ik_residual_m": float(np.linalg.norm(ik_pose - target_xyz)),
                "tracking_error_m": float(np.linalg.norm(measured_pose - ik_pose)),
                "command_duration_s": float(duration),
                "joint_state_sequence": int(result["joint_state_sequence"]),
                "joint_state_received_monotonic": float(
                    result["joint_state_received_monotonic"]
                ),
                "joint_state_stamp_ns": int(result["joint_state_stamp_ns"]),
                "simulation_rate": float(result["simulation_rate"]),
            },
        }