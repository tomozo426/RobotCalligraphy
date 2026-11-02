#!/usr/bin/env python3
"""Fine-tune the shared BC policy with PPO on the training characters."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from bc_train import BehaviorCloningPolicy
from brush_env import BrushCalligraphyEnv, validate_observation_checkpoint
from calligraphy_config import (
    ACTION_SIZE,
    CANVAS_SIZE,
    DEFAULT_DATA_DIR,
    ENDTIP_OFFSET_Z_M,
    HELD_OUT_CHARACTER,
    OBSERVATION_SIZE,
    POLICY_HIDDEN_SIZE,
    PRESSURE_MAPPING_VERSION,
    PROJECT_ROOT,
    TRAIN_CHARACTERS,
)


class PPOPolicy(nn.Module):
    """Tanh-squashed Gaussian actor initialized exactly from the BC MLP."""

    def __init__(self, bc_checkpoint):
        super().__init__()
        bc_policy = BehaviorCloningPolicy(
            observation_size=bc_checkpoint["observation_size"],
            hidden_size=bc_checkpoint["hidden_size"],
            action_size=bc_checkpoint["action_size"],
        )
        bc_policy.load_state_dict(bc_checkpoint["state_dict"])
        self.actor_mean = copy.deepcopy(
            nn.Sequential(*list(bc_policy.network.children())[:-1])
        )
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -1.5))
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, POLICY_HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(POLICY_HIDDEN_SIZE, 1),
        )

    def distribution(self, observations):
        mean = self.actor_mean(observations)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, observations):
        return self.critic(observations).squeeze(-1)

    def deterministic_action(self, observations):
        return torch.tanh(self.actor_mean(observations))


def env_action(normalized_action):
    action = normalized_action.copy()
    action[2] = (action[2] + 1.0) / 2.0
    return action.astype(np.float32)


def collect_episode(env, policy, character, gamma, gae_lambda, max_steps=0):
    observation, _ = env.reset(options={"character": character})
    observations, latent_actions, log_probabilities = [], [], []
    values, rewards = [], []
    motion_diagnostics = []
    final_info = {}
    truncated = False
    while True:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution = policy.distribution(observation_tensor)
            latent_action = distribution.sample()
            log_probability = distribution.log_prob(latent_action).sum(dim=-1)
            value = policy.value(observation_tensor)
        normalized_action = torch.tanh(latent_action).squeeze(0).numpy()
        observation, reward, terminated, truncated, final_info = env.step(
            env_action(normalized_action)
        )
        observations.append(observation_tensor.squeeze(0).numpy())
        latent_actions.append(latent_action.squeeze(0).numpy())
        log_probabilities.append(float(log_probability.item()))
        values.append(float(value.item()))
        rewards.append(float(reward))
        if "joint_state_sequence" in final_info or final_info.get("motion_failed"):
            diagnostic = {
                "step": len(rewards) - 1,
                "motion_failed": bool(final_info.get("motion_failed", False)),
                "motion_failure": final_info.get("motion_failure"),
            }
            for key in (
                "command_xy_px", "sent_xy_px", "measured_xy_px", "tracking_error_px",
                "requested_xyz_m", "ik_fk_xyz_m", "measured_xyz_m",
                "measured_joints_rad",
            ):
                value = final_info.get(key)
                diagnostic[key] = value.tolist() if isinstance(value, np.ndarray) else value
            for key in (
                "post_measurement_valid", "ik_residual_m", "tracking_error_m",
                "command_duration_s", "joint_state_sequence", "joint_state_stamp_ns",
                "joint_state_received_monotonic", "simulation_rate",
                "contact", "pressure", "measured_pressure", "contact_fraction",
            ):
                if key in final_info:
                    diagnostic[key] = final_info[key]
            motion_diagnostics.append(diagnostic)
        if terminated or truncated:
            break
        if max_steps and len(rewards) >= max_steps:
            truncated = True
            break

    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    if truncated:
        with torch.no_grad():
            next_value = float(
                policy.value(
                    torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                ).item()
            )
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * next_value - values[index]
        gae = delta + gamma * gae_lambda * gae
        advantages[index] = gae
        next_value = values[index]
    return {
        "observations": observations,
        "latent_actions": latent_actions,
        "log_probabilities": log_probabilities,
        "values": values,
        "advantages": advantages,
        "returns": advantages + np.asarray(values, dtype=np.float32),
        "reward": float(np.sum(rewards)),
        "dice": float(final_info["dice"]),
        "steps": len(rewards),
        "truncated": truncated,
        "motion_failure": final_info.get("motion_failure"),
        "motion_diagnostics": motion_diagnostics,
    }


def optimize(policy, optimizer, episodes, args):
    observations = torch.as_tensor(
        np.concatenate([episode["observations"] for episode in episodes]), dtype=torch.float32
    )
    latent_actions = torch.as_tensor(
        np.concatenate([episode["latent_actions"] for episode in episodes]), dtype=torch.float32
    )
    old_log_probabilities = torch.as_tensor(
        np.concatenate([episode["log_probabilities"] for episode in episodes]), dtype=torch.float32
    )
    advantages = torch.as_tensor(
        np.concatenate([episode["advantages"] for episode in episodes]), dtype=torch.float32
    )
    returns = torch.as_tensor(
        np.concatenate([episode["returns"] for episode in episodes]), dtype=torch.float32
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = len(observations)
    losses = []
    diagnostics = {key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}

    for _ in range(args.update_epochs):
        for indices in torch.randperm(sample_count).split(args.batch_size):
            distribution = policy.distribution(observations[indices])
            new_log_probabilities = distribution.log_prob(latent_actions[indices]).sum(dim=-1)
            log_ratio = new_log_probabilities - old_log_probabilities[indices]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
            ) * advantages[indices]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(
                policy.value(observations[indices]), returns[indices]
            )
            entropy = distribution.entropy().sum(dim=-1).mean()
            loss = (
                actor_loss
                + args.value_coefficient * value_loss
                - args.entropy_coefficient * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            policy.log_std.data.clamp_(-3.0, 0.5)
            losses.append(float(loss.detach()))
            with torch.no_grad():
                diagnostics["policy_loss"].append(float(actor_loss))
                diagnostics["value_loss"].append(float(value_loss))
                diagnostics["entropy"].append(float(entropy))
                diagnostics["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))
                diagnostics["clip_fraction"].append(float(((ratio - 1).abs() > args.clip_range).float().mean()))

    return float(np.mean(losses)), {key: float(np.mean(values)) for key, values in diagnostics.items()}


def evaluate_deterministic(env, policy, bc_policy, characters, max_steps=0):
    metrics = {}
    for character in characters:
        observation, _ = env.reset(options={"character": character})
        terminated = False
        steps = 0
        while not terminated:
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                bc_action = bc_policy(observation_tensor).squeeze(0).numpy()
            observation, _, terminated, _, info = env.step(bc_action)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        bc_metrics = {
            key: float(info[key])
            for key in ("dice", "overflow_rate", "unpainted_rate")
        }

        observation, _ = env.reset(options={"character": character})
        terminated = False
        steps = 0
        while not terminated:
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                action = policy.deterministic_action(
                    observation_tensor
                ).squeeze(0).numpy()
            action[2] = (action[2] + 1.0) / 2.0
            observation, _, terminated, _, info = env.step(action)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        metrics[character] = {
            "bc": bc_metrics,
            "bc_rl": {
                key: float(info[key])
                for key in ("dice", "overflow_rate", "unpainted_rate")
            },
        }
    return metrics


def safe_lift_motion_node(node):
    joints = node.joint_positions(timeout=2.0)
    pose = node.fk(joints)
    node._raise_tip(pose, joints)


def train(args):
    bc_checkpoint = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
    validate_observation_checkpoint(bc_checkpoint)
    if set(bc_checkpoint["characters"]) != set(TRAIN_CHARACTERS):
        raise ValueError("BC checkpoint must contain exactly the three training characters")
    if HELD_OUT_CHARACTER in bc_checkpoint["characters"]:
        raise ValueError("Held-out character must not enter PPO training")

    if args.motion_backend == "ideal":
        return _train(args, bc_checkpoint, None)

    import rclpy
    from ros2_policy_runner import GakuPolicyRunner

    paper_half_extent = (CANVAS_SIZE - 1) / 2.0 * args.meters_per_pixel
    motion_args = SimpleNamespace(
        texture_topic="",
        require_gripper_controller=False,
        paper_center_x=args.paper_center_x,
        paper_center_y=args.paper_center_y,
        meters_per_pixel=args.meters_per_pixel,
        contact_z=args.contact_z,
        lift_z=args.lift_z,
        pressure_z_span_mm=args.pressure_z_span_mm,
        z_transition_steps=args.z_transition_steps,
        arm_z_offset_mm=args.arm_z_offset_mm,
        command_hz=args.command_hz,
        max_speed_mm_s=args.max_speed_mm_s,
        reset_speed_mm_s=args.reset_speed_mm_s,
        workspace_x_min=args.paper_center_x - paper_half_extent,
        workspace_x_max=args.paper_center_x + paper_half_extent,
        workspace_y_min=args.paper_center_y - paper_half_extent,
        workspace_y_max=args.paper_center_y + paper_half_extent,
    )
    rclpy.init()
    node = None
    try:
        node = GakuPolicyRunner(motion_args, None, None, "feedback", bc_checkpoint)
        return _train(args, bc_checkpoint, node)
    finally:
        if node is not None:
            try:
                safe_lift_motion_node(node)
            except Exception as error:
                print(f"Warning: could not safely lift robot during shutdown: {error}")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _train(args, bc_checkpoint, motion_node):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")
    motion_backend = None
    if motion_node is not None:
        from ros2_feedback import Ros2MotionBackend

        motion_backend = Ros2MotionBackend(motion_node, args.max_speed_mm_s)

    env = BrushCalligraphyEnv(
        artifact_dir=args.data_dir,
        characters=TRAIN_CHARACTERS,
        lock_full_pressure=bc_checkpoint.get("lock_full_pressure", 0.5),
        endpoint_weight=bc_checkpoint.get("endpoint_weight", 1.0),
        path_weight=bc_checkpoint.get("path_weight", 1.0),
        motion_backend=motion_backend,
    )
    if motion_node is not None:
        motion_node.env = env
    if HELD_OUT_CHARACTER in env.character_names:
        raise ValueError("Held-out character must not enter PPO environment")
    policy = PPOPolicy(bc_checkpoint).to(device)

    bc_policy = BehaviorCloningPolicy(
        observation_size=bc_checkpoint["observation_size"],
        hidden_size=bc_checkpoint["hidden_size"],
        action_size=bc_checkpoint["action_size"],
    ).to(device)
    bc_policy.load_state_dict(bc_checkpoint["state_dict"])
    observation_tensor = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
    with torch.no_grad():
        bc_action = bc_policy(observation_tensor)
        ppo_action = policy.deterministic_action(observation_tensor)
        ppo_action[:, 2] = (ppo_action[:, 2] + 1.0) / 2.0
    if not torch.allclose(bc_action, ppo_action, atol=1e-6):
        raise RuntimeError("BC-to-PPO actor transfer changed the deterministic action")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    episode_results = []
    update_losses = []
    update_diagnostics = []
    episode_characters = list(TRAIN_CHARACTERS) * (args.episodes // len(TRAIN_CHARACTERS))
    episode_characters.extend(TRAIN_CHARACTERS[: args.episodes % len(TRAIN_CHARACTERS)])
    rng.shuffle(episode_characters)
    for start in range(0, args.episodes, args.episodes_per_update):
        batch = []
        for character in episode_characters[start : start + args.episodes_per_update]:
            result = collect_episode(
                env, policy, character, args.gamma, args.gae_lambda, args.max_steps
            )
            batch.append(result)
            episode_results.append({
                "character": character,
                "reward": result["reward"],
                "dice": result["dice"],
                "steps": result["steps"],
                "truncated": result["truncated"],
                "motion_failure": result["motion_failure"],
                "motion_diagnostics": result["motion_diagnostics"],
            })
        if motion_node is not None:
            safe_lift_motion_node(motion_node)
        update_loss, diagnostics = optimize(policy, optimizer, batch, args)
        update_losses.append(update_loss)
        update_diagnostics.append(diagnostics)
        print(
            f"update={len(update_losses)} episodes={len(episode_results)}/{args.episodes} "
            f"mean_reward={np.mean([item['reward'] for item in episode_results[-len(batch):]]):.4f} "
            f"mean_dice={np.mean([item['dice'] for item in episode_results[-len(batch):]]):.4f}"
        )

    deterministic_metrics = (
        {} if args.skip_evaluation else evaluate_deterministic(
            env, policy, bc_policy, TRAIN_CHARACTERS, args.max_steps
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **{key: bc_checkpoint[key] for key in (
            "observation_size", "hidden_size", "action_size", "action_convention",
            "observation_convention",
            "characters", "observation_features", "max_xy_offset_px", "control_hz",
            "max_pressure_rate_per_second", "max_brush_width_px",
            "lock_full_pressure",
        ) if key in bc_checkpoint},
        "algorithm": "PPO initialized from BC",
        "motion_backend": args.motion_backend,
        "feedback_source": (
            "/joint_states -> MoveIt FK" if motion_node is not None else "ideal command position"
        ),
        "motion_config": {
            "base_frame": "crane_plus_base",
            "tip_link": "crane_plus_link_endtip",
            "endtip_offset_z_m": ENDTIP_OFFSET_Z_M,
            "arm_joints": [
                "crane_plus_joint1", "crane_plus_joint2",
                "crane_plus_joint3", "crane_plus_joint4",
            ],
            "paper_center_base_m": [args.paper_center_x, args.paper_center_y],
            "meters_per_pixel": args.meters_per_pixel,
            "contact_z_m": args.contact_z,
            "lift_z_m": args.lift_z,
            "pressure_z_span_m": args.pressure_z_span_mm / 1000.0,
            "z_transition_steps": args.z_transition_steps,
            "arm_z_offset_m": args.arm_z_offset_mm / 1000.0,
            "pressure_mapping_version": PRESSURE_MAPPING_VERSION,
            "command_hz": args.command_hz,
            "max_speed_mm_s": args.max_speed_mm_s,
        },
        "source_checkpoint": str(args.bc_checkpoint),
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "evaluation_skipped": args.skip_evaluation,
        "state_dict": policy.state_dict(),
    }
    torch.save(checkpoint, args.output_dir / "bc_rl_policy.pt")
    metrics = {
        "algorithm": "PPO initialized from BC",
        "motion_backend": args.motion_backend,
        "pressure_mapping_version": PRESSURE_MAPPING_VERSION,
        "feedback_source": checkpoint["feedback_source"],
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "evaluation_skipped": args.skip_evaluation,
        "characters": list(TRAIN_CHARACTERS),
        "held_out_character_used_for_training": False,
        "bc_checkpoint": str(args.bc_checkpoint),
        "episode_results": episode_results,
        "deterministic_train_character_metrics": deterministic_metrics,
        "mean_episode_reward": float(np.mean([item["reward"] for item in episode_results])),
        "mean_episode_dice": float(np.mean([item["dice"] for item in episode_results])),
        "training_steps": int(sum(item["steps"] for item in episode_results)),
        "update_losses": update_losses,
        "update_diagnostics": update_diagnostics,
        "final_standard_deviation": policy.log_std.detach().exp().tolist(),
        "device": str(device),
    }
    (args.output_dir / "bc_rl_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "episode_results"}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--bc-checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts" / "day5_day4_self_corner_fixed" / "bc_policy.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "day7_rl",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--motion-backend", choices=("ideal", "gazebo"), default="ideal")
    parser.add_argument("--paper-center-x", type=float, default=0.19)
    parser.add_argument("--paper-center-y", type=float, default=0.0)
    parser.add_argument("--meters-per-pixel", type=float, default=0.0004)
    parser.add_argument("--contact-z", type=float, default=0.0005)
    parser.add_argument("--lift-z", type=float, default=0.0205)
    parser.add_argument("--pressure-z-span-mm", type=float, default=15.0)
    parser.add_argument("--z-transition-steps", type=int, default=3)
    parser.add_argument("--arm-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--command-hz", type=float, default=5.0)
    parser.add_argument("--max-speed-mm-s", type=float, default=15.0)
    parser.add_argument(
        "--reset-speed-mm-s", type=float, default=50.0,
        help="speed for moving to each episode's first lifted task position (default: 50 mm/s)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="truncate each episode after this many steps (0 means full trajectory)",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="skip deterministic BC/PPO rollouts after training",
    )
    args = parser.parse_args()
    minimum_episodes = 1 if args.motion_backend == "gazebo" else len(TRAIN_CHARACTERS)
    if args.episodes < minimum_episodes:
        parser.error(f"episodes must be at least {len(TRAIN_CHARACTERS)}")
    if min(args.episodes_per_update, args.batch_size, args.update_epochs) < 1:
        parser.error("episodes per update, batch size, and update epochs must be positive")
    if args.max_steps < 0:
        parser.error("max steps must be non-negative")
    if args.motion_backend == "gazebo" and min(
        args.meters_per_pixel,
        args.command_hz,
        args.max_speed_mm_s,
        args.pressure_z_span_mm,
    ) <= 0.0:
        parser.error("Gazebo scales, rates, pressure, and lift clearance must be positive")
    if args.motion_backend == "gazebo" and (
        args.lift_z <= args.contact_z
        or args.z_transition_steps < 1
        or not np.isfinite(args.arm_z_offset_mm)
    ):
        parser.error("Gazebo lift height, Z transition steps, and arm Z offset must be valid")
    if args.motion_backend == "gazebo" and args.reset_speed_mm_s is not None and args.reset_speed_mm_s <= 0.0:
        parser.error("--reset-speed-mm-s must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())