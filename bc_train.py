#!/usr/bin/env python3
"""Generate rule-based demonstrations and train a behavior-cloning policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from calligraphy_config import (
    ACTION_SIZE,
    BRUSH_DIRECTION_LOCK_START_PRESSURE,
    CONTROL_HZ,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MULTICHAR_ARTIFACT_DIR,
    HELD_OUT_CHARACTER,
    MAX_PRESSURE_RATE_PER_SECOND,
    MAX_XY_OFFSET_PX,
    OBSERVATION_SIZE,
    POLICY_HIDDEN_SIZE,
    TRAIN_CHARACTERS,
    XY_ACTION_SIZE,
)
from brush_env import BrushCalligraphyEnv, OBSERVATION_CONVENTION


STROKE_START_PRESSURE_SCALES = (0.25, 0.45, 0.70, 1.0)


class BehaviorCloningPolicy(nn.Module):
    """MLP whose tanh outputs map directly to normalized environment actions."""

    def __init__(
        self,
        observation_size=OBSERVATION_SIZE,
        hidden_size=POLICY_HIDDEN_SIZE,
        action_size=ACTION_SIZE,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )

    def forward(self, observations):
        raw_actions = self.network(observations)
        pressure_output = (raw_actions[:, XY_ACTION_SIZE : XY_ACTION_SIZE + 1] + 1.0) / 2.0
        return torch.cat((raw_actions[:, :XY_ACTION_SIZE], pressure_output), dim=1)


def teacher_action(env, step_index):
    """Convert the shared stroke-aware target section into a policy action."""
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    if not env.contact[step_index]:
        return action

    tangent = env._tangent(step_index)
    if not np.any(tangent):
        return action
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    center_offset, stroke_width = env.target_cross_sections[step_index]
    xy_offset = normal * np.clip(center_offset, -env.max_xy_offset_px, env.max_xy_offset_px)
    action[:XY_ACTION_SIZE] = np.clip(xy_offset / env.max_xy_offset_px, -1.0, 1.0)
    action[XY_ACTION_SIZE] = env.pressure_for_brush_width(stroke_width)
    stroke_start = step_index
    while (
        stroke_start > 0
        and env.contact[stroke_start - 1]
        and env.stroke_ids[stroke_start - 1] == env.stroke_ids[step_index]
    ):
        stroke_start -= 1
    local_step = step_index - stroke_start
    if local_step < len(STROKE_START_PRESSURE_SCALES):
        action[XY_ACTION_SIZE] *= STROKE_START_PRESSURE_SCALES[local_step]
    return action


def collect_demonstrations(env, episodes, rng):
    observations = []
    actions = []
    episode_ids = []
    sample_characters = []
    sample_weights = []
    episode_characters = np.resize(np.asarray(env.character_names), episodes)
    rng.shuffle(episode_characters)
    for episode_id in range(episodes):
        character = str(episode_characters[episode_id])
        observation, _ = env.reset(options={"character": character})
        for step_index in range(len(env.trajectory)):
            label = teacher_action(env, step_index)
            observations.append(observation.copy())
            actions.append(label)
            episode_ids.append(episode_id)
            sample_characters.append(character)
            sample_weights.append(env.step_weights[step_index])

            noisy_action = label.copy()
            noisy_action[:XY_ACTION_SIZE] += rng.normal(
                0.0, 0.12, size=XY_ACTION_SIZE
            ).astype(np.float32)
            noisy_action[XY_ACTION_SIZE] += np.float32(rng.normal(0.0, 0.08))
            observation, _, terminated, truncated, _ = env.step(noisy_action)
            if terminated or truncated:
                break

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(episode_ids, dtype=np.int32),
        np.asarray(sample_characters),
        np.asarray(sample_weights, dtype=np.float32),
    )


def save_loss_plot(train_losses, validation_losses, path):
    width, height = 640, 360
    margin = 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot = (margin, margin, width - 20, height - margin)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill="#555555", width=1)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill="#555555", width=1)
    draw.text((margin, 12), "Behavior cloning loss (log scale)", fill="#222222")
    draw.text((width // 2 - 35, height - 20), "Epoch", fill="#222222")
    draw.text((12, height // 2 - 15), "Loss", fill="#222222")
    draw.text((width - 205, 14), "train", fill="#1769aa")
    draw.line((width - 245, 20, width - 212, 20), fill="#1769aa", width=2)
    draw.text((width - 155, 14), "validation", fill="#c45132")
    draw.line((width - 190, 20, width - 162, 20), fill="#c45132", width=2)

    values = np.maximum(np.asarray(train_losses + validation_losses), 1e-8)
    low = float(np.log10(values.min()))
    high = float(np.log10(values.max()))
    if high - low < 0.1:
        high = low + 0.1

    def points(losses):
        result = []
        for index, loss in enumerate(losses):
            x = plot[0] + index * (plot[2] - plot[0]) / max(len(losses) - 1, 1)
            fraction = (np.log10(max(loss, 1e-8)) - low) / (high - low)
            y = plot[3] - fraction * (plot[3] - plot[1])
            result.append((round(x), round(y)))
        return result

    draw.line(points(train_losses), fill="#1769aa", width=2)
    draw.line(points(validation_losses), fill="#c45132", width=2)
    image.save(path)


def evaluate_policy(env, policy, device, character):
    observation, _ = env.reset(options={"character": character})
    terminated = False
    info = {}
    while not terminated:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(tensor).squeeze(0).cpu().numpy()
        observation, _, terminated, _, info = env.step(action)
    return Image.fromarray(env.render()).convert("L"), info


def evaluate_teacher(env, character, animation_path):
    env.reset(options={"character": character})
    terminated = False
    info = {}
    frames = [Image.fromarray(env.render()).convert("RGB")]
    while not terminated:
        action = teacher_action(env, env.step_index)
        _, _, terminated, _, info = env.step(action)
        frames.append(Image.fromarray(env.render()).convert("RGB"))
    frames[0].save(
        animation_path,
        save_all=True,
        append_images=frames[1:],
        duration=25,
        loop=0,
        optimize=False,
    )
    return frames[-1].convert("L"), info


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = BrushCalligraphyEnv(
        artifact_dir=args.artifact_dir,
        max_xy_offset_px=args.max_xy_offset_px,
        control_hz=args.control_hz,
        max_pressure_rate_per_second=args.max_pressure_rate_per_second,
        lock_full_pressure=args.lock_full_pressure,
        endpoint_weight=args.endpoint_weight,
        path_weight=args.path_weight,
    )
    if HELD_OUT_CHARACTER in env.character_names:
        raise ValueError(f"Held-out character {HELD_OUT_CHARACTER!r} must not enter BC training")
    observations, actions, episode_ids, sample_characters, step_weights = collect_demonstrations(
        env, args.episodes, rng
    )
    validation_episodes = []
    for character in env.character_names:
        character_episodes = np.unique(episode_ids[sample_characters == character])
        rng.shuffle(character_episodes)
        validation_count = max(1, int(len(character_episodes) * 0.2))
        validation_episodes.extend(character_episodes[:validation_count])
    validation_episodes = np.asarray(validation_episodes, dtype=np.int32)
    validation_indices = np.flatnonzero(np.isin(episode_ids, validation_episodes))
    training_indices = np.flatnonzero(~np.isin(episode_ids, validation_episodes))

    observation_tensor = torch.from_numpy(observations)
    action_tensor = torch.from_numpy(actions)
    weight_tensor = torch.from_numpy(step_weights)
    dataset = TensorDataset(
        observation_tensor[training_indices],
        action_tensor[training_indices],
        weight_tensor[training_indices],
    )
    training_characters = sample_characters[training_indices]
    character_counts = {
        character: np.count_nonzero(training_characters == character)
        for character in env.character_names
    }
    sample_weights = torch.as_tensor(
        [1.0 / character_counts[character] for character in training_characters],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(training_indices),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    validation_observations = observation_tensor[validation_indices]
    validation_actions = action_tensor[validation_indices]

    policy = BehaviorCloningPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    train_losses = []
    validation_losses = []
    for _ in range(args.epochs):
        policy.train()
        batch_losses = []
        for batch_observations, batch_actions, batch_weights in loader:
            optimizer.zero_grad()
            predictions = policy(batch_observations)
            per_sample = ((predictions - batch_actions) ** 2).mean(dim=1)
            batch_loss = (per_sample * batch_weights).sum() / batch_weights.sum()
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(float(batch_loss.detach()))
        train_losses.append(float(np.mean(batch_losses)))

        policy.eval()
        with torch.no_grad():
            validation_predictions = policy(validation_observations)
            validation_losses.append(float(loss_function(validation_predictions, validation_actions)))

    policy.eval()
    checkpoint = {
        "observation_convention": OBSERVATION_CONVENTION,
        "state_dict": policy.state_dict(),
        "observation_size": OBSERVATION_SIZE,
        "hidden_size": POLICY_HIDDEN_SIZE,
        "action_size": ACTION_SIZE,
        "action_convention": "normalized_xy_offset_and_pressure_target",
        "characters": list(env.character_names),
        "sampling_strategy": "equal_character_weight_with_replacement",
        "observation_features": [
            "actual_xy",
            "reference_xy",
            "normalized_xy_error",
            "pressure",
            "global_progress",
            "stroke_progress",
            "contact",
            "stroke_tangent_xy",
            "target_width_pixels_over_canvas_span",
        ],
        "max_xy_offset_px": env.max_xy_offset_px,
        "control_hz": env.control_hz,
        "max_pressure_rate_per_second": env.max_pressure_rate_per_second,
        "lock_full_pressure": env.lock_full_pressure,
        "endpoint_weight": env.endpoint_weight,
        "path_weight": env.path_weight,
        "max_brush_width_px": env.max_brush_width_px,
    }
    torch.save(checkpoint, output_dir / "bc_policy.pt")
    np.savez_compressed(
        output_dir / "bc_dataset.npz",
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        sample_characters=sample_characters,
        step_weights=step_weights,
    )
    save_loss_plot(train_losses, validation_losses, output_dir / "bc_loss.png")

    character_metrics = {}
    for character in env.character_names:
        teacher_trace, teacher_info = evaluate_teacher(
            env,
            character,
            output_dir / f"bc_teacher_animation_{character}.gif",
        )
        teacher_trace.save(output_dir / f"bc_teacher_trace_{character}.png")
        trace, info = evaluate_policy(env, policy, device, character)
        trace.save(output_dir / f"bc_trace_{character}.png")
        character_metrics[character] = {
            "teacher_dice": teacher_info["dice"],
            "dice": info["dice"],
            "overflow_rate": info["overflow_rate"],
            "unpainted_rate": info["unpainted_rate"],
            "train_episodes": int(np.unique(episode_ids[(sample_characters == character) & ~np.isin(episode_ids, validation_episodes)]).size),
            "validation_episodes": int(np.unique(episode_ids[(sample_characters == character) & np.isin(episode_ids, validation_episodes)]).size),
        }
        for split, indices in (("train", training_indices), ("validation", validation_indices)):
            character_indices = indices[sample_characters[indices] == character]
            with torch.no_grad():
                predictions = policy(observation_tensor[character_indices])
                split_loss = float(loss_function(predictions, action_tensor[character_indices]))
            character_metrics[character][f"{split}_mse"] = split_loss
    metrics = {
        "episodes": args.episodes,
        "samples": len(observations),
        "epochs": args.epochs,
        "initial_train_mse": train_losses[0],
        "final_train_mse": train_losses[-1],
        "initial_validation_mse": validation_losses[0],
        "final_validation_mse": validation_losses[-1],
        "characters": character_metrics,
        "mean_dice": float(np.mean([item["dice"] for item in character_metrics.values()])),
        "control_hz": env.control_hz,
        "max_pressure_rate_per_second": env.max_pressure_rate_per_second,
        "max_brush_width_px": env.max_brush_width_px,
        "device": str(device),
        "seed": args.seed,
    }
    (output_dir / "bc_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MULTICHAR_ARTIFACT_DIR)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-xy-offset-px", type=float, default=MAX_XY_OFFSET_PX)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument(
        "--max-pressure-rate-per-second", type=float, default=MAX_PRESSURE_RATE_PER_SECOND
    )
    parser.add_argument("--lock-full-pressure", type=float, default=0.5)
    parser.add_argument("--endpoint-weight", type=float, default=3.0)
    parser.add_argument("--path-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 2 * len(TRAIN_CHARACTERS) or min(args.epochs, args.batch_size) < 1:
        parser.error(
            f"episodes must be at least {2 * len(TRAIN_CHARACTERS)}; "
            "epochs and batch size must be positive"
        )
    if not BRUSH_DIRECTION_LOCK_START_PRESSURE < args.lock_full_pressure <= 1.0:
        parser.error(
            "lock-full-pressure must be greater than "
            f"{BRUSH_DIRECTION_LOCK_START_PRESSURE} and at most 1.0"
        )
    return args


if __name__ == "__main__":
    train(parse_args())