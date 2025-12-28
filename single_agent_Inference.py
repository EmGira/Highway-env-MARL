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

CHECKPOINT_PATH = os.path.abspath("./checkpoints/2025-12-23/ID_0")  


AGENTS_NR = 2

# Env config
ENV_ID = 'highway-v0'
ENV_CONFIG = {
    "screen_width": 640,
    "screen_height": 480,


    "controlled_vehicles": AGENTS_NR,
    "lanes_count": 4,
    "vehicles_count": 20,
    "reward_speed_range": [40, 50], #increased reward speed
    "collision_reward": -0.5, #incentivise overtakes by reducing punishment on collision


    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
        }
    },

    "action": {
        "type": "MultiAgentAction",
        "action_config": {
        "type": "DiscreteMetaAction",
        }
    },

    "duration": 400,
    "simulation_frequency": 15,
    "ego_spacing": 1,
  
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

    flat_obs_list = [o.flatten() for o in obs]
    flat_obs = np.array(flat_obs_list)

    obs_batch = torch.from_numpy(flat_obs).unsqueeze(0) 
#in un ambiente mulit agente posso creare un batch di piu osservazione
 # qui obs_batch ha shape 1,25   [[obs1]] dove obs_1 e un array di 25 valori
 # potrei creare una batch con tutte le osservazioni degli agenti
 # es: 3 agenti, batch 3, 25 [[obs1], [obs2], [obs3]]
 # in teoria dovrebbe essere comunque compatibile con il modello 

    model_outputs = rl_module.forward_inference({"obs": obs_batch})
    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

    #since our actions are discreete, we simply take the argmax and turn into a tuple 
    best_actions = np.argmax(action_dist_params, axis=1)
    actions_tuple = tuple(a for a in best_actions)
    
    obs, reward, terminated, truncated, info = test_env.step(actions_tuple)
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
