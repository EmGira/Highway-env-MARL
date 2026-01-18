from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

class CrashLoggerCallback(DefaultCallbacks):
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
