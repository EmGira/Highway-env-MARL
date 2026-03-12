import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule   
from ray.rllib.core.rl_module import MultiRLModule 

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback

import highway_env
from highway_env.envs.common.abstract import MultiAgentWrapper

from configs.intersection.IntersectionConfigs import get_multi_agent_config


import pprint 
import os
import torch
import numpy as np
from pathlib import Path

import ray
from ray import tune

ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

CHECKPOINT_PATH = os.path.abspath("./A-checkpoints/2026-03-12/Run_2026-03-12_ID_1/PPO_intersection-v1_multiagent_38496_00000_0_2026-03-12_15-17-36/checkpoint_000006")  


NR_AGENTS = 2
ENV_CONFIG = get_multi_agent_config(num_agents=NR_AGENTS)

multi_rl_module = MultiRLModule.from_checkpoint(
    Path(CHECKPOINT_PATH)
    / "learner_group"
    / "learner"
    / "rl_module"
)


RENDER_MODE = None
sa_env = gym.make("intersection-v1", render_mode = RENDER_MODE, config=ENV_CONFIG)
ma_env = MultiAgentWrapper(sa_env)



NUM_TEST_EPISODES = 20

success_count = 0
crash_count = 0
total_rewards = []

print(f"--- Validation over {NUM_TEST_EPISODES} episodes ---")

for ep in range(NUM_TEST_EPISODES):

    obs, info = ma_env.reset()
    done = [False] * NR_AGENTS
    truncated = False
    ep_reward = 0
    
    while not (all(done) or truncated):
       
        flat_obs = np.stack([o.flatten() for o in obs], axis=0)
        agents_obs = [torch.from_numpy(o).float().unsqueeze(0) for o in flat_obs]
        
       
        with torch.no_grad(): # No need to track operations
            agents_outputs = [multi_rl_module[f"agent_{i}"].forward_inference({"obs": ao}) for ao, i in zip(agents_obs, range(NR_AGENTS))]
            agents_actions = [torch.argmax(out["action_dist_inputs"], dim=1).item() for out in agents_outputs]

       
        obs, reward, done, truncated, info = ma_env.step(tuple(agents_actions))
        ep_reward += sum(reward)
        

        
        if RENDER_MODE == "human":
            ma_env.render()

    
    is_crashed = info.get('crashed', False)
    is_success = all(done) and not is_crashed

    if is_success:
        success_count += 1
    if is_crashed:
        crash_count += 1

    
    
    total_rewards.append(ep_reward)
    print(f"Episodio {ep+1}: {'SUCCESS' if is_success else 'FAILED'} | Reward: {ep_reward:.2f}")



success_rate = (success_count / NUM_TEST_EPISODES) * 100
avg_reward = np.mean(total_rewards)

print("\n" + "="*30)
print(f"Results ({NUM_TEST_EPISODES} episodes)")
print(f"Success Rate: {success_rate}% ")
print(f"Crash Rate:   {(crash_count/NUM_TEST_EPISODES)*100}% ")
print(f"Average Reward: {avg_reward:.2f}")
print("="*30)