import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule   
from ray.tune.registry import register_env
import highway_env
import os
import torch
import numpy as np
import ray

ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

CHECKPOINT_PATH = os.path.abspath("./checkpoints/2025-12-19/ID_0")  

ENV_ID = "highway-v0"
ENV_CONFIG = {
    "screen_width": 640,
    "screen_height": 480,

    "controlled_vehicles": 2,
    "lanes_count": 4,
    "vehicles_count": 20,
    "reward_speed_range": [30, 40], #increased reward speed
    "collision_reward": -1,

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


def env_creator(env_config):
    import highway_env
    return gym.make(ENV_ID, render_mode="rgb_array", config=env_config)

register_env(ENV_ID, env_creator)



trained_algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
rl_module = trained_algo.get_module()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rl_module.to(device)



print("\n--- Avvio Rendering del modello addestrato su nuovo enviroment di test ---")

test_env = gym.make(ENV_ID, render_mode="human", config=ENV_CONFIG)
obs, info = test_env.reset()
done = truncated = False
episode_return = 0

while not (done or truncated):
  
    combined_obs = np.concatenate([o.flatten() for o in obs])

    obs_batch = torch.from_numpy(combined_obs).unsqueeze(0).float()
    obs_batch = obs_batch.to(device)

    model_outputs = rl_module.forward_inference({"obs":obs_batch})

    all_logits = model_outputs["action_dist_inputs"][0]
    print("model_output: ", model_outputs)
    print("All logits: ", all_logits)

    
    logits_agent_1 = all_logits[:5]
    logits_agent_2 = all_logits[5:]

    # 3. Trova l'indice del valore massimo (l'azione) per ogni agente
    action_1 = torch.argmax(logits_agent_1).item()
    action_2 = torch.argmax(logits_agent_2).item()
    
    greedy_action = (action_1, action_2)
    
    selected_actions = [ACTIONS_ALL[a] for a in greedy_action]
    print(f"GREEDY ACTION: {selected_actions}")
 
 
    obs, reward, done, truncated, info = test_env.step(greedy_action)
    test_env.render()
    print("INFO", info)
    print(f"REWARD: {reward}")

    episode_return += np.sum(reward)

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
