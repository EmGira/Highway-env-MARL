import torch
import torch.nn as nn
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
import gymnasium as gym

class CentralizedCriticSACModel(SACTorchModel):
    """Custom model implementing Centralized Critic for MASAC.
    
    The Actor (policy) uses only local observations to compute actions.
    The Critic (Q-network) uses the global state + action to compute Q-values.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 policy_model_config=None, q_model_config=None, twin_q=False,
                 initial_alpha=1.0, target_entropy=None):
        
        super().__init__(obs_space, action_space, num_outputs, model_config, name,
                         policy_model_config, q_model_config, twin_q, initial_alpha, target_entropy)
        
        # 1. Dimension Extraction
        if hasattr(obs_space, "original_space") and isinstance(obs_space.original_space, gym.spaces.Dict):
            self.local_obs_dim = obs_space.original_space["obs"].shape[0]
            self.global_state_dim = obs_space.original_space["global_state"].shape[0]
        elif isinstance(obs_space, gym.spaces.Dict):
            self.local_obs_dim = obs_space["obs"].shape[0]
            self.global_state_dim = obs_space["global_state"].shape[0]
        else:
            self.local_obs_dim = model_config.get("custom_model_config", {}).get("local_obs_dim", 27)
            self.global_state_dim = obs_space.shape[0] - self.local_obs_dim

        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.discrete:
            self.action_dim = action_space.n
            self.q_out_dim = self.action_dim
            self.q_in_extra = 0
        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.q_out_dim = 1
            self.q_in_extra = self.action_dim
        else:
            self.action_dim = 1
            self.q_out_dim = 1
            self.q_in_extra = 1

        # 2. Re-build Actor (Action Model) -> Maps Local Obs to Actions
        hiddens_actor = policy_model_config.get("fcnet_hiddens", [256, 256]) if policy_model_config else [256, 256]
        act_activation = policy_model_config.get("fcnet_activation", "relu") if policy_model_config else "relu"
        act_layer = nn.ReLU if act_activation == "relu" else nn.Tanh
        
        actor_layers = []
        in_dim = self.local_obs_dim
        for h in hiddens_actor:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(act_layer())
            in_dim = h
        
        # SAC requires the output to be `num_outputs` (usually 2 * action_dim for mean and log_std)
        actual_num_outputs = num_outputs
        if actual_num_outputs is None:
            if self.discrete:
                actual_num_outputs = self.action_dim
            else:
                actual_num_outputs = self.action_dim * 2
                
        actor_layers.append(nn.Linear(in_dim, actual_num_outputs))
        self.action_model = nn.Sequential(*actor_layers)

        # 3. Re-build Critic (Q-Network) -> Maps Global State + Action to Q-Value
        hiddens_q = q_model_config.get("fcnet_hiddens", [256, 256]) if q_model_config else [256, 256]
        q_activation = q_model_config.get("fcnet_activation", "relu") if q_model_config else "relu"
        q_layer = nn.ReLU if q_activation == "relu" else nn.Tanh
        
        def build_q_net():
            q_layers = []
            in_dim = self.global_state_dim + self.q_in_extra
            for h in hiddens_q:
                q_layers.append(nn.Linear(in_dim, h))
                q_layers.append(q_layer())
                in_dim = h
            q_layers.append(nn.Linear(in_dim, self.q_out_dim))
            return nn.Sequential(*q_layers)

        self.q_net = build_q_net()
        if twin_q:
            self.twin_q_net = build_q_net()

    def forward(self, input_dict, state, seq_lens):
       
        obs_input = input_dict["obs"]
        if isinstance(obs_input, dict):
            local_obs = obs_input["obs"]
            global_state = obs_input["global_state"]
        else:
            local_obs = obs_input[:, :self.local_obs_dim]
            global_state = obs_input[:, self.local_obs_dim:]

        return torch.cat([local_obs, global_state], dim=-1), state

    def get_action_model_outputs(self, model_out):
        local_obs = model_out[:, :self.local_obs_dim]
        return self.action_model(local_obs), []

    def _extract_global_state(self, model_out):
        gs = model_out[:, self.local_obs_dim:]
        if gs.shape[-1] > self.global_state_dim:
            gs = gs[:, :self.global_state_dim]
        elif gs.shape[-1] < self.global_state_dim:
            padding = torch.zeros(gs.shape[0], self.global_state_dim - gs.shape[-1], device=gs.device)
            gs = torch.cat([gs, padding], dim=-1)
        return gs

    def get_q_values(self, model_out, actions=None):
        gs = self._extract_global_state(model_out)
        if self.discrete:
            return self.q_net(gs), []
        else:
            return self.q_net(torch.cat([gs, actions], -1)), []

    def get_twin_q_values(self, model_out, actions=None):
        gs = self._extract_global_state(model_out)
        if self.discrete:
            return self.twin_q_net(gs), []
        else:
            return self.twin_q_net(torch.cat([gs, actions], -1)), []

    def policy_variables(self, as_dict: bool = False):
        if as_dict:
            return {k: v for k, v in self.action_model.named_parameters()}
        return list(self.action_model.parameters())

    def q_variables(self, as_dict: bool = False):
        q_vars = {k: v for k, v in self.q_net.named_parameters()} if as_dict else list(self.q_net.parameters())
        if hasattr(self, "twin_q_net") and self.twin_q_net:
            twin_q_vars = {k: v for k, v in self.twin_q_net.named_parameters()} if as_dict else list(self.twin_q_net.parameters())
            if as_dict:
                q_vars.update(twin_q_vars)
            else:
                q_vars += twin_q_vars
        return q_vars
