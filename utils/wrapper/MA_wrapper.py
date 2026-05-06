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

        obs_cfg = config.get("observation", {}).get("observation_config", {})
        self._is_absolute = obs_cfg.get("absolute", True)
        
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

        self._obs_space_in_preferred_format = True #TODOO try removing these
        self._action_space_in_preferred_format = True



    def _process_obs(self, agent_obs_matrix):

        if self._is_absolute:
            return agent_obs_matrix.flatten().astype(np.float32)

        rel_obs = agent_obs_matrix.copy()
        
        # True only where the vehicle is present (presence = 1)
        present_mask = rel_obs[:, 0] == 1.0

        # extract all features 
        # 1=x, 2=y, 3=vx, 4=vy, 5=cos_h, 6=sin_h
        ego_x = rel_obs[0, 1]
        ego_y = rel_obs[0, 2]
        ego_vx = rel_obs[0, 3]
        ego_vy = rel_obs[0, 4]
        ego_cos = rel_obs[0, 5]
        ego_sin = rel_obs[0, 6]

        # 3. TRASLATION (relative space)
        #subtract position features to all present vehicles
        rel_obs[present_mask, 1] -= ego_x
        rel_obs[present_mask, 2] -= ego_y
        rel_obs[present_mask, 3] -= ego_vx
        rel_obs[present_mask, 4] -= ego_vy

        # 4. ROTATION (Ego-Centric Frame)
        # extract traslated positions
        dx = rel_obs[present_mask, 1].copy()
        dy = rel_obs[present_mask, 2].copy()
        dvx = rel_obs[present_mask, 3].copy()
        dvy = rel_obs[present_mask, 4].copy()
        
    
        # extract cos e sin of present vehicles
        other_cos = rel_obs[present_mask, 5].copy()
        other_sin = rel_obs[present_mask, 6].copy()

       
        # Apply inverse rotation matrix
        
        # rotated positions
        rel_obs[present_mask, 1] = dx * ego_cos + dy * ego_sin
        rel_obs[present_mask, 2] = -dx * ego_sin + dy * ego_cos
        
        # rotated speeds
        rel_obs[present_mask, 3] = dvx * ego_cos + dvy * ego_sin
        rel_obs[present_mask, 4] = -dvx * ego_sin + dvy * ego_cos
        
        # rotated heading
        # cos(A - B) = cos(A)cos(B) + sin(A)sin(B)
        # sin(A - B) = sin(A)cos(B) - cos(A)sin(B)
        rel_obs[present_mask, 5] = other_cos * ego_cos + other_sin * ego_sin
        rel_obs[present_mask, 6] = other_sin * ego_cos - other_cos * ego_sin

        return rel_obs.flatten().astype(np.float32)


    def reset(self, *, seed=None, options=None):
        self._terminated_agents = set()

        obs, info = self.env.reset(seed=seed, options=options)
        
        flat_obs = {}

        for i, agent_id in enumerate(self._agent_list):
            flat_obs[agent_id] = self._process_obs(obs[i])


            

        
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
            

                obs_dict[agent_id] = self._process_obs(obs[i])
    

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



