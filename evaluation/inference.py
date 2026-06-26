import gymnasium as gym
import highway_env


import pprint 
import torch
import numpy as np

import ray
from ray import tune
from ray.rllib.core.rl_module import MultiRLModule, RLModule

from pathlib import Path
import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper

from configs.intersection.IntersectionConfigs import get_simple_multi_agent_config, get_improved_Simple_config, get_ego_only_config


from utils.evaluation_utils import load_policy_or_module, create_eval_env, compute_actions_stochastic, compute_actions_stochastic_legacy, compute_actions, compute_actions_legacy

from src.models.CentralizedCriticModel import CentralizedCriticModel
from src.models.CentralizedCriticSACModel import CentralizedCriticSACModel

from ray.rllib.models import ModelCatalog


ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)
ModelCatalog.register_custom_model("centralized_critic_sac_model", CentralizedCriticSACModel)



CHECKPOINT_PATH = os.path.abspath(
    "./A-checkpoints/TEST/3agents-MAPPO/MAPPO_2/ID_a2e6e_00000/checkpoint_000005"
    )  


NR_AGENTS = 3
ENV_CONFIG = get_ego_only_config(num_agents=NR_AGENTS)

ENV_CONFIG["simulation_frequency"] = 15
ENV_CONFIG["randomize_controlled_vehicles"] = False


model_or_policy, stack_type = load_policy_or_module(CHECKPOINT_PATH)


RENDER_MODE = "human"
ma_env = create_eval_env(stack_type, ENV_CONFIG, "customRoundabout-env-v0", render_mode=RENDER_MODE, inference_mode=True)




NUM_TEST_EPISODES = 50

success_count = 0
crash_count = 0
total_rewards = []

print(f"--- Validation over {NUM_TEST_EPISODES} episodes ---")

for ep in range(NUM_TEST_EPISODES):

    obs, info = ma_env.reset()
   
    all_agent_ids = list(obs.keys())

    done = {"__all__": False}
    truncated = {"__all__": False}

    ep_reward = 0
    
    while not (done["__all__"] or truncated["__all__"]):
       
        
        if stack_type == "new":
            agents_actions = compute_actions(model_or_policy, obs)
        else:
            agents_actions = compute_actions_legacy(model_or_policy, obs)
        
       
        obs, reward, done, truncated, info = ma_env.step(agents_actions)
        # print(reward)
    
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