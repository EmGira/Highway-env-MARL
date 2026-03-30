#https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html#custom-callbacks
#https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.html#ray.rllib.callbacks.callbacks.RLlibCallback

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

       
        all_infos = episode.get_infos()  # {agent_id: [info]}
      
        overall_success = 0
        overall_crashed = 0

        for agent_id, infos in all_infos.items():
                if not infos:
                    continue

               
                if any(info.get("all_arrived", False) for info in infos):
                    overall_success = 1
                    
                if any(info.get("crashed", False) for info in infos):
                    overall_crashed = 1


                speeds = [info["speed"] for info in infos if "speed" in info]
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    metrics_logger.log_value(f"Custom/average_speed_{agent_id}", avg_speed)

        metrics_logger.log_value("Custom/success_rate", overall_success)
        metrics_logger.log_value("Custom/crash_incident_rate", overall_crashed)
        
        # Pulizia manuale della memoria GPU
        if hasattr(self, "empty_cache"): # Se hai importato torch
            empty_cache()
    
    empty_cache()


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
        