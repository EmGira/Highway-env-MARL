import gymnasium as gym
import highway_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RLlibHighwayWrapper(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.env = gym.make("highway-v0", config=config)
        self.possible_agents = [f"agent_{i+1}" for i in range(config["controlled_vehicles"])]
        self.agents = self.possible_agents

    
        self.observation_space = {
            agent_id: self.env.observation_space[i] 
            for i, agent_id in enumerate(self.agents)
        }
        
        self.action_space = {
            agent_id: self.env.action_space[i] 
            for i, agent_id in enumerate(self.agents)
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        return {self.agents[i]: obs[i] for i in range(len(obs))}, info

    def step(self, action_dict):

        actions = tuple(action_dict[agent_id] for agent_id in self.agents)
        
        obs, reward, terminated, truncated, info = self.env.step(actions)

        obs_dict = {self._agent_ids[i]: obs[i] for i in range(len(obs))}
        rew_dict = {self._agent_ids[i]: reward[i] for i in range(len(reward))}
        
     
        term_dict = {agent_id: terminated for agent_id in self._agent_ids}
        term_dict["__all__"] = terminated
        
        trunc_dict = {agent_id: truncated for agent_id in self._agent_ids}
        trunc_dict["__all__"] = truncated

        return obs_dict, rew_dict, term_dict, trunc_dict, info

    def render(self):
        return self.env.render()
    

ENV_CONFIG = {
    "screen_width": 640,
    "screen_height": 480,

    "controlled_vehicles": 3,
    "lanes_count": 4,
    "vehicles_count": 20,
    "reward_speed_range": [40, 50], 
    "collision_reward": -0.5, 

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


env = RLlibHighwayWrapper(ENV_CONFIG)

print("AGENT IDs: ", env.agents)
print("ACTION SPACE: ", env.action_spaces)
print("OBS SPACE: ",env.action_spaces)


obs, info = env.reset()
print("OBS: ", obs)
print("INFO: ", info)


