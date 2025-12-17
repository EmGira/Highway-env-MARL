import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule   
from ray.tune.registry import register_env
import highway_env
import os
import torch
import numpy as np
import ray


CHECKPOINT_PATH = os.path.abspath("./checkpoints/2025-12-16_ID_0")  

ENV_ID = "highway-v0"
HIGHWAY_ENV_CONFIG = {
  "lanes_count": 2,
    "observation": {
        "type": "Kinematics"
    },
    "duration": 100
}

def env_creator(env_config):
    import highway_env
    return gym.make(ENV_ID, render_mode="rgb_array", config=env_config)

register_env(ENV_ID, env_creator)



trained_algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
rl_module = trained_algo.get_module()
#rl_module = RLModule.from_checkpoint(
#    CHECKPOINT_PATH
#)


print("\n--- Avvio Rendering del modello addestrato su nuovo enviroment di test ---")

test_env = gym.make(ENV_ID, render_mode="human", config=HIGHWAY_ENV_CONFIG)
obs, info = test_env.reset()
done = truncated = False
episode_return = 0
while not (done or truncated):
    
    # Compute the next action from a batch (B=1) of observations.
    flat_obs = obs.flatten()

    obs_batch = torch.from_numpy(flat_obs).float().unsqueeze(0)

    model_outputs = rl_module.forward_inference({"obs":obs_batch})
    
    action_dist_params = model_outputs["action_dist_inputs"][0].numpy()

    # For discrete actions, take the argmax over the logits:
    greedy_action = np.argmax(action_dist_params)

    obs, reward, done, truncated, info = test_env.step(greedy_action)
    test_env.render()

    episode_return += reward

print("Episode return:", episode_return)    
test_env.close()


#close open ray processes
trained_algo.stop()
ray.shutdown()



# For CONTINOUS ACTIONS, take mean (max likelyhood action)
   # greedy_action = np.clip(
   #     action_dist_params[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
   #     a_min=test_env.action_space.low[0],
   #     a_max=test_env.action_space.high[0],
   # )
