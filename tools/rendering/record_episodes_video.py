
import os
import sys
import numpy as np
import ray
import imageio
from ray.rllib.models import ModelCatalog

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_folder)

from utils.evaluation_utils import (
    load_policy_or_module,
    create_eval_env,
    compute_actions,
    compute_actions_legacy,
)
from configs.intersection.IntersectionConfigs import get_ego_only_config

from src.models.CentralizedCriticModel import CentralizedCriticModel
from src.models.CentralizedCriticSACModel import CentralizedCriticSACModel

ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)
ModelCatalog.register_custom_model("centralized_critic_sac_model", CentralizedCriticSACModel)

#./A-checkpoints/MASAC_fail_1m/MASAC_0/ID_01a10_00000/checkpoint_000013
CHECKPOINT_PATH = os.path.abspath(
    "./A-checkpoints/MASAC_fail_1m/MASAC_0/ID_01a10_00000/checkpoint_000013"
)
ENV_NAME = "customIntersection-env-v0" 
NUM_AGENTS = 5
NUM_EPISODES = 10
BASE_SEED = 100  
FPS = 30
OUTPUT_PATH = os.path.abspath("./thesis_media/MASAC_Intersection_5agents_10episodes.mp4")


def record_episodes_video():
    ENV_CONFIG = get_ego_only_config(num_agents=NUM_AGENTS)
    ENV_CONFIG["simulation_frequency"] = 15
    ENV_CONFIG["randomize_controlled_vehicles"] = False

    model_or_policy, stack_type = load_policy_or_module(CHECKPOINT_PATH)

    ma_env = create_eval_env(
        stack_type, ENV_CONFIG, ENV_NAME, render_mode="rgb_array", inference_mode=True
    )
    ma_env.env.env.metadata["render_fps"] = FPS

    frames = []

    for ep in range(NUM_EPISODES):
        seed = BASE_SEED + ep
        obs, info = ma_env.reset(seed=seed)

        frame = ma_env.render()
        if frame is not None:
            frames.append(frame)

        done = {"__all__": False}
        truncated = {"__all__": False}
        ep_reward = 0.0

        while not (done["__all__"] or truncated["__all__"]):
            if stack_type == "new":
                agents_actions = compute_actions(model_or_policy, obs)
            else:
                agents_actions = compute_actions_legacy(model_or_policy, obs)

            obs, reward, done, truncated, info = ma_env.step(agents_actions)
            ep_reward += sum(reward.values())

            frame = ma_env.render()
            if frame is not None:
                frames.append(frame)

        last_info = list(info.values())[0] if info else {}
        is_crashed = last_info.get("crashed", False)
        is_success = last_info.get("all_arrived", False)
        outcome = "SUCCESS" if is_success else ("CRASHED" if is_crashed else "TRUNCATED")
        print(f"Ep {ep + 1}/{NUM_EPISODES} (seed={seed}): {outcome} | Reward: {ep_reward:.2f}")

    ma_env.close()

    if not frames:
        print("Nessun frame catturato: rendering fallito.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimsave(OUTPUT_PATH, frames, fps=FPS, macro_block_size=1)
    print(f"\nVideo salvato: {OUTPUT_PATH} ({len(frames)} frame, {FPS} fps)")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    record_episodes_video()
    ray.shutdown()
