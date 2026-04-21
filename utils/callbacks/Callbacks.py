#https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html#custom-callbacks
#https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.html#ray.rllib.callbacks.callbacks.RLlibCallback

import math

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from torch.cuda import empty_cache
import statistics

class CrashLoggerCallback(DefaultCallbacks): #TODOO change to new API stack RLlibCallback
    def on_episode_start(self, *, episode: MultiAgentEpisode, **kwargs):
        pass
        
    def on_episode_step(self, *, episode: MultiAgentEpisode, **kwargs):
        pass



    def on_episode_end(self, *, episode: MultiAgentEpisode, metrics_logger=None, **kwargs):    
        if not metrics_logger:
            return

       
        all_infos = episode.get_infos()  # {agent_1: [info, ...], agent_2: [info, ...]}
      
        overall_success = 0
        overall_crashed = 0

        for agent_id, infos in all_infos.items():
                
                if not infos:
                    continue
                
                last_info = infos[-1]
                if last_info.get("crashed", False):
                    overall_crashed = 1
                elif last_info.get("all_arrived", False):
                    overall_success = 1


                agent_idx = int(agent_id.split("_")[1])
                speeds = [info["speed"][agent_idx] for info in infos if "speed" in info]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    metrics_logger.log_value(f"Custom/average_speed_{agent_id}", avg_speed)


        metrics_logger.log_value("Custom/success_rate", overall_success)
        metrics_logger.log_value("Custom/crash_incident_rate", overall_crashed)
    

        #Log metrics to be used to calculate STD on_train_result
        ep_return = episode.get_return()
        
      
        metrics_logger.log_value("Custom/ep_return_mean", ep_return, reduce="mean")
        metrics_logger.log_value("Custom/ep_return_sq_mean", ep_return ** 2, reduce="mean")


        if hasattr(self, "empty_cache"):
            empty_cache()
    
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class SafeEvaluationCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):

        if not hasattr(algorithm, "_last_eval_score"):
            algorithm._last_eval_score = 0.0 

        #Return Mean to be used as metric for optuna and scheduler (same as ep_return_mean, but is = 0 before a evaluation occurs)
        if "evaluation" in result and "env_runners" in result["evaluation"]:
            eval_data = result["evaluation"]["env_runners"]
            
   
            if "episode_return_mean" in eval_data:
                val = eval_data["episode_return_mean"]
                
                if val is not None and not math.isnan(val):
                    algorithm._last_eval_score = val

        
        result["safe_return_mean"] = algorithm._last_eval_score

        def calculate_std(mean_x, mean_x2):
            # Var = E[X^2] - (E[X])^2
            variance = mean_x2 - (mean_x ** 2)
            std = math.sqrt(max(0.0, variance))
            return std
        
        #STD DEV for train and eval
        if "env_runners" in result:
            runners = result["env_runners"]
            
            if "Custom/ep_return_mean" in runners and "Custom/ep_return_sq_mean" in runners:
                std = calculate_std(runners["Custom/ep_return_mean"], runners["Custom/ep_return_sq_mean"])
                runners["Custom/episode_return_std"] = std
                
                

        if "evaluation" in result and "env_runners" in result["evaluation"]:
            eval_runners = result["evaluation"]["env_runners"]

            if "Custom/ep_return_mean" in eval_runners and "Custom/ep_return_sq_mean" in eval_runners:
                std = calculate_std(runners["Custom/episode_return_std"], eval_runners["Custom/ep_return_sq_mean"])
                eval_runners["Custom/episode_return_std"] = std



#https://github.com/ray-project/ray/issues/51560#issuecomment-2758195710 thread for AdamBetas Fix
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from torch import Tensor

class FixAdamBetasCallback(DefaultCallbacks):
    

    def on_checkpoint_loaded(self, *, algorithm, **kwargs) -> None:
         
        def betas_tensor_to_float(learner):
            for param_grp_key in learner._optimizer_parameters.keys():
                param_grp = param_grp_key.param_groups[0]
                param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])

                if "betas" in param_grp and isinstance(param_grp["betas"][0], Tensor):
                    param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
                    
        
        algorithm.learner_group.foreach_learner(betas_tensor_to_float)
        