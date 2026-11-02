#!/usr/bin/env python3
"""Display live joint positions from a ROS 2 JointState topic."""

import argparse
import threading

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState


RADIANS_PER_DEGREE = 3.14159 / 180.0
SERVO_HOME_RAD = 150.0 * RADIANS_PER_DEGREE
JOINT_LIMITS_RAD = {
    "crane_plus_joint1": (
        0.0 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
        300.0 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
    ),
    "crane_plus_joint2": (
        45.45 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
        252.20 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
    ),
    "crane_plus_joint3": (
        3.52 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
        290.62 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
    ),
    "crane_plus_joint4": (
        44.57 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
        251.32 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
    ),
    "crane_plus_joint_hand": (
        109.38 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
        188.27 * RADIANS_PER_DEGREE - SERVO_HOME_RAD,
    ),
}


class JointPositionMonitor(Node):
    def __init__(self, topic, arm_state_topic, gripper_state_topic):
        super().__init__("joint_position_monitor")
        self.latest_positions = None
        self.latest_targets = {}
        self.create_subscription(JointState, topic, self._on_joint_state, 10)
        self.create_subscription(
            JointTrajectoryControllerState,
            arm_state_topic,
            self._on_controller_state,
            10,
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            gripper_state_topic,
            self._on_controller_state,
            10,
        )

    def _on_joint_state(self, message):
        self.latest_positions = dict(zip(message.name, message.position))

    def _on_controller_state(self, message):
        self.latest_targets.update(zip(message.joint_names, message.reference.positions))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument(
        "--arm-state-topic",
        default="/crane_plus_arm_controller/controller_state",
    )
    parser.add_argument(
        "--gripper-state-topic",
        default="/crane_plus_gripper_controller/controller_state",
    )
    parser.add_argument("--refresh-ms", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_ms <= 0:
        raise ValueError("--refresh-ms must be positive")

    rclpy.init()
    node = JointPositionMonitor(args.topic, args.arm_state_topic, args.gripper_state_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    figure, axis = plt.subplots(figsize=(9, 4.5))
    figure.canvas.manager.set_window_title("ROS 2 Joint Positions")

    def update(_frame):
        axis.clear()
        axis.axvline(0.0, color="#555555", linewidth=1)
        axis.set_xlim(-3.4, 3.4)
        axis.set_xticks(range(-3, 4))
        axis.set_xlabel("Position (rad)")
        axis.set_title("Joint positions: actual and target")
        axis.grid(axis="x", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)

        if not node.latest_positions:
            axis.text(
                0.5,
                0.5,
                f"Waiting for JointState on {args.topic}...",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_yticks([])
            return

        names = list(node.latest_positions)
        positions = [node.latest_positions[name] for name in names]
        for index, name in enumerate(names):
            limits = JOINT_LIMITS_RAD.get(name)
            if limits is not None:
                lower, upper = limits
                axis.hlines(index, lower, upper, color="#b7c9ce", linewidth=7, zorder=1)
                axis.vlines(
                    [lower, upper],
                    index - 0.32,
                    index + 0.32,
                    color="#c2413b",
                    linewidth=1.7,
                    zorder=4,
                )
        axis.barh(
            names,
            positions,
            color="#168c83",
            height=0.36,
            label="Actual",
            zorder=2,
        )
        for index, position in enumerate(positions):
            axis.annotate(
                f"A {position:+.3f}",
                (position, index),
                xytext=(5 if position >= 0 else -5, -7),
                textcoords="offset points",
                ha="left" if position >= 0 else "right",
                va="center",
                fontsize=9,
            )
        target_names = [name for name in names if name in node.latest_targets]
        target_values = [node.latest_targets[name] for name in target_names]
        target_indices = [names.index(name) for name in target_names]
        axis.scatter(
            target_values,
            target_indices,
            marker="D",
            color="#e08b26",
            edgecolors="white",
            linewidths=0.6,
            s=38,
            zorder=5,
            label="Target",
        )
        for name, target, index in zip(target_names, target_values, target_indices):
            axis.annotate(
                f"T {target:+.3f}",
                (target, index),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#9a5b0a",
            )
        axis.legend(
            handles=[
                Patch(color="#168c83", label="Actual position"),
                Line2D(
                    [0],
                    [0],
                    color="none",
                    marker="D",
                    markerfacecolor="#e08b26",
                    markeredgecolor="white",
                    label="Target position",
                ),
                Line2D([0], [0], color="#b7c9ce", linewidth=7, label="Allowed range"),
                Line2D(
                    [0],
                    [0],
                    color="#c2413b",
                    marker="|",
                    linestyle="None",
                    markersize=12,
                    label="Lower / upper limit",
                ),
            ],
            loc="lower right",
            fontsize=8,
        )

    animation = FuncAnimation(
        figure,
        update,
        interval=args.refresh_ms,
        cache_frame_data=False,
    )
    try:
        plt.show()
    finally:
        animation.event_source.stop()
        executor.shutdown()
        ros_thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()