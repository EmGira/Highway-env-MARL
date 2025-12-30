import os
from tqdm import trange

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

import gymnasium as gym
import highway_env

import glob
from pathlib import Path
import datetime


today = datetime.date.today()

if not os.path.isdir(f"./checkpoints/{today.strftime('%Y-%m-%d')}"):
    os.mkdir(f"./checkpoints/{today.strftime('%Y-%m-%d')}")

checkpoints_dir = Path(f"./checkpoints/{today.strftime('%Y-%m-%d')}")
nr_of_subdirectories = len([f for f in checkpoints_dir.iterdir() if f.is_dir()])

# Env config
ENV_ID = 'highway-v0'

ENV_CONFIG = {

    "observation": {
        "type": "Kinematics"
        },
    "action": {
        "type": "DiscreteMetaAction",
        },

    "lanes_count": 4,
    "vehicles_count": 50,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 40, 
    "ego_spacing": 2, 
    "vehicles_density": 1, #try 2


    "collision_reward": -5,  
    "right_lane_reward": 0.1,  #try increasing
    "high_speed_reward": 0.4,  
    "lane_change_reward": 0,  

    "reward_speed_range": [20, 30],

    "normalize_reward": True,
    "offroad_terminal": False,

}

ray.init(
        log_to_driver=False, configure_logging=True,
        logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO")
        )

def env_creator(env_config):
    import highway_env
    return gym.make(ENV_ID, render_mode=None, config=env_config)

tune.register_env(ENV_ID, env_creator)


# Algorithm config
config = (
    SACConfig()
    .environment(
        env=ENV_ID,
        env_config=ENV_CONFIG
    )
    .framework("torch")

    .env_runners(
        num_env_runners=8,
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
        num_envs_per_env_runner=2,
        rollout_fragment_length="auto"
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=5,
        evaluation_config=SACConfig.overrides(
            explore=False, 
            env_config=ENV_CONFIG
        ) #config usato per l'env in fase di valutazione
    )

    .training(
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=3e-4,

        # evita picchi ne gradiente quando lagente viene punto severamente (es: schianto)
        grad_clip=1.0,
        grad_clip_by="global_norm",

        train_batch_size_per_learner=1024, 
        
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 50000,  # Conserva 50000 esperienze diverse
            "alpha": 0.6, #peso sulla priorità
        },

    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )

)   


algo = config.build_algo()
log_dir = algo.logdir

# Training
for i in range(100):
    result = algo.train()
       
    training_stats = result.get('env_runners', result.get('sampler_results', {}))
    training_reward = training_stats.get('episode_return_mean', float('nan'))
    

    evaluation_stats = result.get('evaluation', {})
    evaluation_reward = evaluation_stats.get('env_runners', {}).get('episode_return_mean', 0)
    
    print(f"Iter {i+1} | Train Reward: {training_reward:.2f} | Eval Reward: {evaluation_reward:.2f}")

algo.evaluate()



path = f"./checkpoints/{today.strftime('%Y-%m-%d')}/ID_{nr_of_subdirectories}"
saved_results = algo.save(checkpoint_dir = os.path.abspath(path))
checkpoint_dir = saved_results.checkpoint.path

if checkpoint_dir:
    print(f"\n@@@ Checkpoint salvato correttamente!")
    print(f"@@@ Percorso: {checkpoint_dir}\n")


algo.stop()



train_files = glob.glob(os.path.join(log_dir, "result.json"))

if train_files:
    print(f"@@@ I risultati di addestramento sono salvati in: {train_files[0]} \n")
    print(f"execute: tensorboard --logdir={train_files[0]} to see results") 
else:
    print("\n$$$ Attenzione: il file result.json non è stato trovato.")