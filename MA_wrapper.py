import gymnasium as gym
import highway_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RLlibHighwayWrapper(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.env = gym.make("highway-v0", config=config)
        self.possible_agents = [f"agent_{i+1}" for i in range(config["controlled_vehicles"])]
        self.agents = self.possible_agents

    
        self.observation_spaces = gym.spaces.Dict({
            agent_id: self.env.observation_space[i] 
            for i, agent_id in enumerate(self.agents)
        })
        
        self.action_spaces = gym.spaces.Dict({
            agent_id: self.env.action_space[i] 
            for i, agent_id in enumerate(self.agents)
        })

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        return {self.agents[i]: obs[i] for i in range(len(obs))}, info

    def step(self, action_dict):

        actions = tuple(action_dict[agent_id] for agent_id in self.agents)
        
        obs, reward, terminated, truncated, info = self.env.step(actions)

        obs_dict = {self.agents[i]: obs[i] for i in range(len(obs))}
        rew_dict = {self.agents[i]: reward[i] for i in range(len(reward))}
        
     
        term_dict = {agent_id: terminated for agent_id in self.agents}
        term_dict["__all__"] = terminated
        
        trunc_dict = {agent_id: truncated for agent_id in self.agents}
        trunc_dict["__all__"] = truncated

        return obs_dict, rew_dict, term_dict, trunc_dict, {}

    def render(self):
        return self.env.render()


