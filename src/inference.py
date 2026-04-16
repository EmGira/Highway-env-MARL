import gymnasium as gym
import highway_env


import pprint 
import torch
import numpy as np

import ray
from ray import tune
from ray.rllib.core.rl_module import MultiRLModule 

from pathlib import Path
import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper

from configs.intersection.IntersectionConfigs import get_busy_intersection_config, get_simple_multi_agent_config


def compute_actions(multi_rl_module, obs):

    policy_module = multi_rl_module["shared_policy"]

    with torch.no_grad():
        agents_actions = {}
        for agent_id, agent_obs in obs.items():
       
            ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
            output = policy_module.forward_inference({"obs": ao})
            agents_actions[agent_id] = torch.argmax(output["action_dist_inputs"], dim=1).item()

            
    
    return agents_actions

CHECKPOINT_PATH = os.path.abspath(
    "./A-checkpoints/SimpleConfig_B_500iter/Run0/PPO_Batch_2048-lr_2.375e-05_ID_d2aae_00000/checkpoint_000038"
    )  


NR_AGENTS = 2
ENV_CONFIG = get_simple_multi_agent_config(num_agents=NR_AGENTS)
ENV_CONFIG["simulation_frequency"] = 15

# ENV_CONFIG["spawn_points"] = ["0", "1"]
# ENV_CONFIG["multi_destinations"] = ["o1", "o0"]

# ENV_CONFIG["spawn_points"] = ["3", "1"]
# ENV_CONFIG["multi_destinations"] = ["o0", "o3"]

ENV_CONFIG["spawn_points"] = ["2", "1"] 
ENV_CONFIG["multi_destinations"] = ["o3", "o3"] 


multi_rl_module = MultiRLModule.from_checkpoint(
    Path(CHECKPOINT_PATH)
    / "learner_group"
    / "learner"
    / "rl_module"
)


RENDER_MODE = "human"
ma_env = RLlibHighwayWrapper(config=ENV_CONFIG, env_id="customIntersection-env-v0", render_mode=RENDER_MODE)




NUM_TEST_EPISODES = 20

success_count = 0
crash_count = 0
total_rewards = []

print(f"--- Validation over {NUM_TEST_EPISODES} episodes ---")

for ep in range(NUM_TEST_EPISODES):

    obs, info = ma_env.reset()

    done = {"__all__": False}
    truncated = {"__all__": False}

    ep_reward = 0
    
    while not (done["__all__"] or truncated["__all__"]):
       
        agents_actions = compute_actions(multi_rl_module, obs)

       
        obs, reward, done, truncated, info = ma_env.step(agents_actions)
        ep_reward += sum(reward.values())

        if RENDER_MODE != None:
            ma_env.render()

        # print("@@info:")
        # pprint.pprint(info)

   
    last_info = list(info.values())[0] if info else {}
    
    is_crashed = last_info.get('crashed', False)
  
    is_success = last_info.get("all_arrived", False)

    if is_success:
        success_count += 1
    if is_crashed:
        crash_count += 1

    
    total_rewards.append(ep_reward)
    print(f"Ep {ep+1}: {'SUCCESS' if is_success else ('CRASHED' if is_crashed else 'TRUNCATED')} | Reward: {ep_reward:.2f}")



success_rate = (success_count / NUM_TEST_EPISODES) * 100
avg_reward = np.mean(total_rewards)

print("\n" + "="*30)
print(f"Results ({NUM_TEST_EPISODES} episodes)")
print(f"Success Rate: {success_rate}% ")
print(f"Crash Rate:   {(crash_count/NUM_TEST_EPISODES)*100}% ")
print(f"Average Reward: {avg_reward:.2f}")
print("="*30)

ma_env.close()