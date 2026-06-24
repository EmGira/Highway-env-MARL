import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import gymnasium as gym

class CentralizedCriticModel(TorchModelV2, nn.Module):
    """Custom model implementing Centralized Training and Decentralized Execution (MAPPO).
    
    The Actor (policy) uses only local observations to compute actions during execution.
    The Critic (value function) uses the global state (concatenation of all agents' observations) during training.

    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

      
        if hasattr(obs_space, "original_space") and isinstance(obs_space.original_space, gym.spaces.Dict):

            self.local_obs_dim = obs_space.original_space["obs"].shape[0]
            self.global_state_dim = obs_space.original_space["global_state"].shape[0]


        elif isinstance(obs_space, gym.spaces.Dict):

            self.local_obs_dim = obs_space["obs"].shape[0]
            self.global_state_dim = obs_space["global_state"].shape[0]


        else:

            self.local_obs_dim = model_config.get("custom_model_config", {}).get("local_obs_dim", 27)
            self.global_state_dim = obs_space.shape[0] - self.local_obs_dim


        #based on ray rl lib default model:
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "tanh")

        if activation == "tanh":
            act_layer = nn.Tanh
        elif activation == "relu":
            act_layer = nn.ReLU
        else:
            act_layer = nn.Tanh


        #Actor network: maps local obs -> action logits
        actor_layers = []
        in_dim = self.local_obs_dim

        for h in hiddens:

            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(act_layer())
            in_dim = h

        actor_layers.append(nn.Linear(in_dim, num_outputs))
        self.actor = nn.Sequential(*actor_layers)


        #Critic network: maps global state -> value estimate
        critic_layers = []
        in_dim = self.global_state_dim

        for h in hiddens:

            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(act_layer())
            in_dim = h

        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)


        self._value_out = None


    def forward(self, input_dict, state, seq_lens):

        obs_input = input_dict["obs"]

        if isinstance(obs_input, dict):

            obs = obs_input["obs"]
            global_state = obs_input["global_state"]

        else:

            obs = obs_input[:, :self.local_obs_dim]
            global_state = obs_input[:, self.local_obs_dim:]



        # Compute policy logits (decentralized execution)
        logits = self.actor(obs)

        # Ensure global_state matches self.global_state_dim to prevent dimension mismatches when evaluating with nr of agents greater than in training.
        if global_state.shape[-1] > self.global_state_dim:

            global_state = global_state[:, :self.global_state_dim]


        elif global_state.shape[-1] < self.global_state_dim:

            padding = torch.zeros(global_state.shape[0], self.global_state_dim - global_state.shape[-1], device=global_state.device)
            global_state = torch.cat([global_state, padding], dim=-1)



        # Compute centralized critic value (centralized training)
        self._value_out = self.critic(global_state)


        return logits, state


    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return torch.reshape(self._value_out, [-1])