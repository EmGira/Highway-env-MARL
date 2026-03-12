#https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html#custom-callbacks
#https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.html#ray.rllib.callbacks.callbacks.RLlibCallback

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

class CrashLoggerCallback(DefaultCallbacks): #TODOO change to new API stack RLlibCallback
    def on_episode_start(self, *, episode: MultiAgentEpisode, **kwargs):
        episode.custom_data["n_crashes"] = 0

    def on_episode_step(self, *, episode: MultiAgentEpisode, **kwargs):

        agent_rewards = episode.get_rewards()

        for reward_list in agent_rewards.values():
           for r in reward_list:
                
                if r <= -4.0: 
                    episode.custom_data["n_crashes"] += 1


    def on_episode_end(self, *, episode: MultiAgentEpisode, metrics_logger=None, **kwargs):    

        crashes = episode.custom_data.get("n_crashes", 0)
        
        if metrics_logger:
            metrics_logger.log_value("crashes_total", crashes)


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