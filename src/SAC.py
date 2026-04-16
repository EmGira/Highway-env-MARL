from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback
from configs.intersection.IntersectionConfigs import get_multi_agent_config
import highway_env

from ray import tune
from ray.rllib.algorithms.sac import SACConfig

import os
import glob
from pathlib import Path
import datetime

today = datetime.date.today()

if not os.path.isdir(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}"):
    os.mkdir(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}")

checkpoints_dir = Path(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}")
nr_of_subdirectories = len([f for f in checkpoints_dir.iterdir() if f.is_dir()])



#CONFIG
NR_AGENTS = 2
ENV_CONFIG = get_multi_agent_config(NR_AGENTS)
ENV_CONFIG["duration"] = 200


tune.register_env("intersection-v1_multiagent", lambda config: RLlibHighwayWrapper(ENV_CONFIG, "intersection-v1")) #TODOO.1 add envID as parameter

config = (
    SACConfig()
    .environment(
        env = "intersection-v1_multiagent"
    )
    .framework("torch")
    .env_runners(
        num_env_runners=4,  #engaging 4 gpu cores
        num_envs_per_env_runner=1, #each core simulating 2 envs

        #env_to_module_connector=lambda env, spaces, device: FlattenObservations(), this caused issues, flattening happens in the MA_wrapper now
        sample_timeout_s=200.0,
        rollout_fragment_length="auto" 
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=5,
        evaluation_duration=20,
        evaluation_duration_unit="episodes",

    )
    .training(
         #uses two Q-networks for value action estimation, to experiment with in future
        n_step=1,

        train_batch_size = 256,
        num_steps_sampled_before_learning_starts = 2000,

        lr=None,
        #default values!
        actor_lr=3e-5,
        critic_lr=3e-4,
        alpha_lr=3e-4,

        replay_buffer_config={
            "type": "MultiAgentEpisodeReplayBuffer",
            "capacity": 50000,
        }
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )
    .multi_agent(
        policies={"agent_0", "agent_1"}, 
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id, 
        count_steps_by="env_steps"
    )
    .callbacks(CrashLoggerCallback)
)

algo = config.build_algo()
log_dir = algo.logdir
print("@@@ Logging directory: ", log_dir)



#TRAINING
for i in range(100):
    result = algo.train()

    print(f"Iterazione {i + 1}:")
    print(f"\tRicompensa Media (Train): {result['env_runners']['episode_return_mean']:.2f}")        #average total reward (sum of all agents rewards) 
    print(f"\tLunghezza Media (Train): {result['env_runners']['episode_len_mean']:.2f}")            #average total nr of steps per episode (until all agents terminated)
    print(f"\tRicompensa Media per agente (Train): {result['env_runners']['agent_episode_returns_mean']}")
    print(f"\tNr Passi per agente (Train): {result['env_runners']['agent_steps']}")



#EVALUATION AND SHUTDOWN
algo.evaluate()

path = f"./A-checkpoints/{today.strftime('%Y-%m-%d')}/ID_{nr_of_subdirectories}"
saved_results = algo.save(checkpoint_dir = os.path.abspath(path))
checkpoint_dir = saved_results.checkpoint.path

if checkpoint_dir:
    print(f"\n@@@ Checkpoint salvato correttamente!")
    print(f"\t Percorso: {checkpoint_dir}\n")

algo.stop()



train_files = glob.glob(os.path.join(log_dir, "result.json"))

if train_files:
    print(f"\n@@@ I risultati di addestramento sono salvati in: {log_dir}")
    print(f"\texecute: tensorboard --logdir={log_dir} to see results") 
else:
    print("\n$$$ Attenzione: il file result.json non è stato trovato.")