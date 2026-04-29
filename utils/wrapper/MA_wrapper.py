#https://docs.ray.io/en/latest/rllib/package_ref/env/multi_agent_env.html#multi-agent-env-reference-docs

import gymnasium as gym
import highway_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from highway_env.envs.common.abstract import MultiAgentWrapper
import numpy as np

class RLlibHighwayWrapper(MultiAgentEnv):
    def __init__(self, config, env_id, render_mode = None): 
        super().__init__()
        sa_env = gym.make(env_id, render_mode=render_mode, config=config) #"intersection-v1"
        self.env = MultiAgentWrapper(sa_env)
        
        self._agent_list = [f"agent_{i}" for i in range(config["controlled_vehicles"])]
        self._agent_ids = set(self._agent_list)

         #TODOO deletable
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

        self._obs_space_in_preferred_format = True #TODOO try removing these
        self._action_space_in_preferred_format = True


    def reset(self, *, seed=None, options=None):
        self._terminated_agents = set()

        obs, info = self.env.reset(seed=seed, options=options)
        
        # obs are flattened in the wrapper (flattenobservations() in the algorithm config caused errors)
        flat_obs = {
            agent_id: obs[i].flatten().astype(np.float32)
            for i, agent_id in enumerate(self._agent_list)
        }
        
        return flat_obs, {agent_id: info for agent_id in self._agent_list}


    def step(self, action_dict):
        # Tracking agents that terminated
        if not hasattr(self, '_terminated_agents'):
            self._terminated_agents = set()
        
    
        if not action_dict:
            raise ValueError("action_dict is empty or None!")

        # build tuple of the actions of every agent
        actions = []
        for agent_id in self._agent_list:
            if agent_id in action_dict:
                actions.append(action_dict[agent_id])
            else:
                actions.append(4) #default action when terminated
        
        actions = tuple(actions)
        
        
        obs, rewards, dones, truncated, info = self.env.step(actions)
    
        obs_dict = {}
        rew_dict = {}
        term_dict = {}
        trunc_dict = {}
        info_dict = {}
        
       
        is_trunc_iterable = isinstance(truncated, (list, tuple, np.ndarray))
        is_info_iterable = isinstance(info, (list, tuple, np.ndarray))
        
        for i, agent_id in enumerate(self._agent_list):
            if agent_id not in self._terminated_agents:                                              
                obs_dict[agent_id] = obs[i].flatten()
                rew_dict[agent_id] = rewards[i]
                term_dict[agent_id] = dones[i]
                
                
                agent_truncated = truncated[i] if is_trunc_iterable else truncated
                trunc_dict[agent_id] = agent_truncated
                
                info_dict[agent_id] = info[i] if is_info_iterable else info

           
                agent_done = dones[i]
                agent_trunc = truncated[i] if is_trunc_iterable else truncated
                
                if agent_done or agent_trunc:
                    self._terminated_agents.add(agent_id)
        
        #episoded end only when all agents are terminated
        is_all_done = len(self._terminated_agents) == len(self._agent_list)
        
        term_dict["__all__"] = is_all_done
        trunc_dict["__all__"] = False 
        
        if is_all_done:
            self._terminated_agents = set()
            
        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict


    def render(self):
        return self.env.render()

    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        if hasattr(super(), 'close'):
            super().close()



