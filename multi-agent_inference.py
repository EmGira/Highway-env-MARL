import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule   
from ray.rllib.core.rl_module import MultiRLModule 

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback

import highway_env
from highway_env.envs.common.abstract import MultiAgentWrapper



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

CHECKPOINT_PATH = os.path.abspath("./checkpoints/2026-01-18/ID_5")  


NR_AGENTS = 2
ENV_CONFIG = {

    "observation": { 
        "type": "MultiAgentObservation",
        "observation_config": { "type": "Kinematics" }
    },
    "action": { 
        "type": "MultiAgentAction",
        "action_config": { "type": "DiscreteMetaAction" }
    },

 
    "controlled_vehicles": NR_AGENTS,
    "vehicles_count": 10, 

    "reward_speed_range": [10, 30], 
    
 
    "collision_reward": -5.0, 
    
   
    "high_speed_reward": 1, 
    "arrived_reward": 5.0, 

    "on_road_reward": 0.1,
    "offroad_terminal": True,
    "reward_config": {
         "collision_reward": -5.0, 
    },
    
    
    "normalize_reward": False, 
    
    "duration": 200, 
}



multi_rl_module = MultiRLModule.from_checkpoint(
    Path(CHECKPOINT_PATH)
    / "learner_group"
    / "learner"
    / "rl_module"
)



print("\n--- Avvio Rendering del modello addestrato su nuovo enviroment di test ---")

sa_env = gym.make("intersection-v1", render_mode = "rgb_array", config=ENV_CONFIG)
ma_env = MultiAgentWrapper(sa_env)

obs, info = ma_env.reset()
done = truncated = False

episode_rewards = [0.0] * NR_AGENTS

while not (done or truncated):
    
    if isinstance(obs, (tuple, list)):
        flat_obs = np.stack([o.flatten() for o in obs], axis=0)
        is_multi_agent = True
    else: #TODOO in this case the check can be removed as this will always be a list of observations
        flat_obs = obs.flatten()[np.newaxis, :] #we must add a dimention so that Ray accepts the obs
        is_multi_agent = False



    agents_obs = [torch.from_numpy(o).float().unsqueeze(0) for o in flat_obs]
    agents_outputs = [multi_rl_module[f"agent_{i}"].forward_inference({"obs": agent_o}) for agent_o, i in zip(agents_obs, range(NR_AGENTS))]
    agents_actions = [torch.argmax(out["action_dist_inputs"], dim=1) for out in agents_outputs] 

    action_set = tuple(a for a in agents_actions)
    

    
    obs, reward, done, truncated, info = ma_env.step(action_set)
    ma_env.render()
    
    for i in range(NR_AGENTS):
        episode_rewards[i] += reward[i]


print(f"Total rewards per agent: {episode_rewards}")
print(f"Total episode return: {sum(episode_rewards)}")

ma_env.close()


ray.shutdown()


