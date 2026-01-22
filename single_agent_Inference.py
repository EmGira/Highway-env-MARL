import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule   
from ray.tune.registry import register_env
import highway_env
import os
import torch
import numpy as np
import ray
from pathlib import Path

CHECKPOINT_PATH = os.path.abspath("./checkpoints/2025-12-30/ID_7")  


AGENTS_NR = 1

# Env config
ENV_ID = 'highway-v0'
ENV_CONFIG = {

    "observation": {
        "type": "Kinematics"
        },
    "action": {
        "type": "DiscreteMetaAction"
        },

    "lanes_count": 4,
    "vehicles_count": 50,
    "controlled_vehicles": AGENTS_NR,
    "initial_lane_id": None,
    "duration": 40, 
    "ego_spacing": 2,
    "vehicles_density": 1, 


    "collision_reward": -1, 
    "right_lane_reward": 0.1,  
    "high_speed_reward": 0.4,
    "lane_change_reward": 0,

    "reward_speed_range": [20, 30],

    "normalize_reward": True,
    "offroad_terminal": False
}

#    "observation": {
#        "type": "Kinematics"
#   },
#   "action": {
#       "type": "DiscreteMetaAction",
#   },




def env_creator(env_config):
    import highway_env
    return gym.make(ENV_ID, render_mode="rgb_array", config=env_config)

register_env(ENV_ID, env_creator)



#trained_algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
#rl_module = trained_algo.get_module()
#rl_module = RLModule.from_checkpoint(
#    CHECKPOINT_PATH
#)
rl_module = RLModule.from_checkpoint(
    Path(CHECKPOINT_PATH)
    / "learner_group"
    / "learner"
    / "rl_module"
    / "default_policy"
)


print("\n--- Avvio Rendering del modello addestrato su nuovo enviroment di test ---")

test_env = gym.make(ENV_ID, render_mode="human", config=ENV_CONFIG)
obs, info = test_env.reset()
terminated = truncated = False
episode_return = 0

while not (terminated or truncated):
    


    # Compute the next action from a batch (B=1) of observations.
    if isinstance(obs, (tuple, list)):
        flat_obs = np.stack([o.flatten() for o in obs])
        is_multi_agent = True
    else:
        flat_obs = obs.flatten()[np.newaxis, :] #we must add a dimention so that Ray accepts the obs
        is_multi_agent = False


    obs_batch = torch.from_numpy(flat_obs).unsqueeze(0) 
    model_outputs = rl_module.forward_inference({"obs": obs_batch})
    action_dist_params = model_outputs["action_dist_inputs"][0].detach().numpy() #add .detach() if the algo requires grad. (ex: SAC)

    #since our actions are discreete, we simply take the argmax and turn into a tuple 
    best_actions = np.argmax(action_dist_params, axis=1)
    if(is_multi_agent):
        action_set = tuple(a for a in best_actions)
    else:
        action_set = best_actions
    
    obs, reward, terminated, truncated, info = test_env.step(action_set)
    test_env.render()

    episode_return += reward

print("Episode return:", episode_return)    
test_env.close()


#close open ray processes
#trained_algo.stop()
ray.shutdown()



# For CONTINOUS ACTIONS, take mean (max likelyhood action)
   # greedy_action = np.clip(
   #     action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
   #     a_min=test_env.action_space.low[0],
   #     a_max=test_env.action_space.high[0],
   # )
