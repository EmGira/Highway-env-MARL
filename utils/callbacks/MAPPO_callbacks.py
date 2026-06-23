import math
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from torch import Tensor

class MAPPOCrashLoggerCallback(DefaultCallbacks):
    """Legacy-compatible crash logging callback for MAPPO."""
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        # Accumulate speeds during the episode
        agents = episode.get_agents()
        for agent_id in agents:
            last_info = episode.last_info_for(agent_id)
            if last_info and "speed" in last_info:
                agent_idx = int(agent_id.split("_")[1])
                speeds_list = episode.user_data.setdefault(f"speeds_{agent_id}", [])
                speeds_list.append(last_info["speed"][agent_idx])

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        agents = list(episode.get_agents())
        
        overall_success = 0
        overall_crashed = 0
        crash_speeds = []
        num_agents = len(agents)

        for agent_id in agents:
            last_info = episode.last_info_for(agent_id)
            if not last_info:
                continue

            agent_idx = int(agent_id.split("_")[1])
            if last_info.get("crashed", False):
                overall_crashed = 1
                if "speed" in last_info:
                    crash_speeds.append(last_info["speed"][agent_idx])
            elif last_info.get("all_arrived", False):
                overall_success = 1

            # Retrieve average speed from user_data
            speeds = episode.user_data.get(f"speeds_{agent_id}", [])
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                episode.custom_metrics[f"Custom/average_speed_{agent_id}"] = avg_speed

        episode.custom_metrics["Custom/success_rate"] = overall_success
        episode.custom_metrics["Custom/crash_incident_rate"] = overall_crashed

        episode.custom_metrics[f"Custom/Density_{num_agents}/success_rate"] = overall_success
        episode.custom_metrics[f"Custom/Density_{num_agents}/crash_rate"] = overall_crashed

        if crash_speeds:
            avg_crash_speed = sum(crash_speeds) / len(crash_speeds)
            episode.custom_metrics["Custom/speed_at_impact"] = avg_crash_speed


class MAPPOSafeEvaluationCallback(DefaultCallbacks):
    """Legacy-compatible safe evaluation callback for MAPPO."""
    def on_train_result(self, *, algorithm, result, **kwargs):
        if not hasattr(algorithm, "_last_eval_score"):
            algorithm._last_eval_score = 0.0 

        # Return Mean to be used as metric for optuna and scheduler
        if "evaluation" in result:
            eval_data = result["evaluation"]
            
            # Check new API stack key
            if "env_runners" in eval_data and "episode_return_mean" in eval_data["env_runners"]:
                val = eval_data["env_runners"]["episode_return_mean"]
            # Check legacy API stack key
            elif "episode_reward_mean" in eval_data:
                val = eval_data["episode_reward_mean"]
            else:
                val = None
                
            if val is not None and not math.isnan(val):
                algorithm._last_eval_score = val

        result["safe_return_mean"] = algorithm._last_eval_score


class MAPPOFixAdamBetasCallback(DefaultCallbacks):
    """Fix for Adam betas tensor issue when loading checkpoints."""
    def on_checkpoint_loaded(self, *, algorithm, **kwargs) -> None:
        def betas_tensor_to_float(learner):
            for param_grp_key in learner._optimizer_parameters.keys():
                param_grp = param_grp_key.param_groups[0]
                param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])

                if "betas" in param_grp and isinstance(param_grp["betas"][0], Tensor):
                    param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
                    
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)
