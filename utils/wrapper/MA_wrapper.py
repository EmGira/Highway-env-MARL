#https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html#multi-agent-env-reference-docs

import gymnasium as gym
import highway_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from highway_env.envs.common.abstract import MultiAgentWrapper
import numpy as np

class RLlibHighwayWrapper(MultiAgentEnv):
    def __init__(self, config): #TODOO.1 add envID as parameter
        super().__init__()
        sa_env = gym.make("intersection-v1", config=config)
        self.env = MultiAgentWrapper(sa_env)

        
        self._agent_list = [f"agent_{i}" for i in range(config["controlled_vehicles"])]
        self._agent_ids = set(self._agent_list)
        self.agents = self._agent_ids
        self.possible_agents = self._agent_ids
    
        # compute the observation space of a single agent
        original_obs_space = self.env.observation_space[0]
        if len(original_obs_space.shape) > 1:
     
            flat_dim = int( np.prod(original_obs_space.shape) )
            single_agent_obs_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(flat_dim,), 
                dtype=np.float32
            )
        else:
            single_agent_obs_space = original_obs_space
        

        #turn obs_space and action_space into dictionary, wich is the format requested by RLlib for MA-envs
        self.observation_space = gym.spaces.Dict({
            agent_id: single_agent_obs_space
            for agent_id in self._agent_list
        })
        self.action_space = gym.spaces.Dict({
            agent_id: self.env.action_space[i] 
            for i, agent_id in enumerate(self._agent_list)
        })

        self._obs_space_in_preferred_format = True #TODOO try removing these, shouldn't be an issue
        self._action_space_in_preferred_format = True


    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # obs are flattened in the wrapper (flattenobservations() in the algorithm config caused errors)
        flat_obs = {
            agent_id: obs[i].flatten() 
            for i, agent_id in enumerate(self._agent_list)
        }
        
        return flat_obs, {agent_id: info for agent_id in self._agent_list}


    def step(self, action_dict):
        # Tracking agents that terminated: 
        # we need to do this because RLlib requires that agents that have terminated shouldnt return their reward
        if not hasattr(self, '_terminated_agents'):
            self._terminated_agents = set()
        
        if not hasattr(self, '_step_count'): #TODOO, can be removed,
            self._step_count = 0
        self._step_count += 1
        
        
        if not action_dict:
            raise ValueError("action_dict is empty or None!")
        

        # build tuple of the actions of every agent (for those who terminated, default to 0)
        actions = []
        for agent_id in self._agent_list:

            if agent_id in action_dict:
                actions.append(action_dict[agent_id])

            elif agent_id in self._terminated_agents:
                actions.append(0)

            else:
                print(f"  WARNING: {agent_id} not in action_dict and not terminated!")
                actions.append(0)
        
        actions = tuple(actions)
        
        #execute actions in the enviroment and compute the results in the format required by RayRLlib
        obs, rewards, dones, truncated, info = self.env.step(actions)

        
        obs_dict = {}
        rew_dict = {}
        term_dict = {}
        
        for i, agent_id in enumerate(self._agent_list):
            
            
            # return data only for non-terminated Agents or agents that terminated in the current step: this is required by RayRLlib
            if agent_id not in self._terminated_agents:                                               
                obs_dict[agent_id] = obs[i].flatten()
                rew_dict[agent_id] = rewards[i]
                term_dict[agent_id] = dones[i]

            # add agent to the terminated list
            if dones[i]:
                self._terminated_agents.add(agent_id)
            
        
        # Truncated and Info: returned only for active agents
        trunc_dict = {agent_id: truncated for agent_id in obs_dict.keys()}
        trunc_dict["__all__"] = truncated
    
        info_dict = {agent_id: info for agent_id in obs_dict.keys()}

        
        
        # __all__ is true when all agents have completed the episode
        term_dict["__all__"] = all(dones)
        
        if term_dict["__all__"]:
            self._terminated_agents = set()
            

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict


    def render(self):
        return self.env.render()



