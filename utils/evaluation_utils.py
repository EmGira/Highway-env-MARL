import os
import sys
import torch
import numpy as np
from pathlib import Path
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.policy.policy import Policy

def load_policy_or_module(checkpoint_path):
    """Loads either a MultiRLModule (New API Stack) or Policy (Old API Stack) from a checkpoint path."""

    from utils.models.CentralizedCriticModel import CentralizedCriticModel
    from ray.rllib.models import ModelCatalog
    
    
    ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)
    
    new_stack_path = Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    if new_stack_path.exists():
        # New API Stack
        return MultiRLModule.from_checkpoint(new_stack_path), "new"
    else:
        # Old API Stack
        legacy_path = Path(checkpoint_path) / "policies" / "shared_policy"
        if legacy_path.exists():
            return Policy.from_checkpoint(os.path.abspath(legacy_path)), "old"
        else:
            raise FileNotFoundError(f"Neither new stack path nor legacy path found in {checkpoint_path}")

def create_eval_env(stack_type, env_config, env_id, render_mode=None, inference_mode=False):
    """Instantiates the correct environment wrapper (MAPPO_wrapper for MAPPO, MA_wrapper for IPPO)."""
    if stack_type == "old":
        from utils.wrapper.MAPPO_wrapper import RLlibMAPPOHighwayWrapper
        return RLlibMAPPOHighwayWrapper(config=env_config, env_id=env_id, render_mode=render_mode, inference_mode=inference_mode)
    else:
        from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
        return RLlibHighwayWrapper(config=env_config, env_id=env_id, render_mode=render_mode, inference_mode=inference_mode)


#NEW API action compute:
def compute_actions(multi_rl_module, obs):
    policy_module = multi_rl_module["shared_policy"]
    with torch.no_grad():
        agents_actions = {}
        for agent_id, agent_obs in obs.items():
            ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
            output = policy_module.forward_inference({"obs": ao})
            agents_actions[agent_id] = torch.argmax(output["action_dist_inputs"], dim=1).item()
    return agents_actions

def compute_continous_actions(multi_rl_module, obs, env_agent_ids):
    policy_module = multi_rl_module["shared_policy"]
    with torch.no_grad():
        agents_actions = {}
        for agent_id in env_agent_ids:
            if agent_id in obs:
                agent_obs = obs[agent_id]
                ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
                output = policy_module.forward_inference({"obs": ao})
                action_dist_params = output["action_dist_inputs"][0].cpu().numpy()
                greedy_action = np.clip(
                    action_dist_params[0:1], 
                    a_min=-1.0,
                    a_max=1.0,
                )
                agents_actions[agent_id] = greedy_action
            else:
                agents_actions[agent_id] = np.array([0.0], dtype=np.float32)
    return agents_actions


#OLD API action compute:
def compute_actions_legacy(policy, obs, explore=False):
    """Computes actions for each agent using a legacy Policy."""
    agents_actions = {}
    for agent_id, agent_obs in obs.items():
        agents_actions[agent_id] = policy.compute_single_action(agent_obs, explore=explore)[0]
    return agents_actions

def compute_continous_actions_legacy(policy, obs, env_agent_ids, explore=False):
    """Computes continuous actions for each agent using a legacy Policy."""
    agents_actions = {}
    for agent_id in env_agent_ids:
        if agent_id in obs:
            action = policy.compute_single_action(obs[agent_id], explore=explore)[0]
            agents_actions[agent_id] = np.clip(action, -1.0, 1.0)
        else:
            agents_actions[agent_id] = np.array([0.0], dtype=np.float32)
    return agents_actions


#Discrete Action Compute with exploratiuon
def compute_actions_stochastic(multi_rl_module, obs):
    policy_module = multi_rl_module["shared_policy"]
    with torch.no_grad():
        agents_actions = {}
        for agent_id, agent_obs in obs.items():
            ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
            output = policy_module.forward_inference({"obs": ao})
            logits = output["action_dist_inputs"]
            dist = torch.distributions.Categorical(logits=logits)
            agents_actions[agent_id] = dist.sample().item()
    return agents_actions

def compute_actions_stochastic_legacy(policy, obs):
    agents_actions = {}
    for agent_id, agent_obs in obs.items():
        agents_actions[agent_id] = policy.compute_single_action(agent_obs, explore=True)[0]
    return agents_actions
