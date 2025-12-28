import sys
import os
from tqdm import tqdm

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.train import RunConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # Questo viene eseguito ogni volta che una macchina finisce il percorso
        reward = episode.get_return()
        length = episode.length
        print(f" -> Episodio finito: Reward={reward:.2f}, Lunghezza={length}")

import gymnasium as gym
import highway_env

import glob
from pathlib import Path
import datetime

pbar = tqdm(range(10), desc="Training RLlib")

today = datetime.date.today()

if not os.path.isdir(f"./checkpoints/{today.strftime('%Y-%m-%d')}"):
    os.mkdir(f"./checkpoints/{today.strftime('%Y-%m-%d')}")

checkpoints_dir = Path(f"./checkpoints/{today.strftime('%Y-%m-%d')}")
nr_of_subdirectories = len([f for f in checkpoints_dir.iterdir() if f.is_dir()])

# Env config
ENV_ID = 'highway-v0'

ENV_CONFIG = {
    "screen_width": 640,
    "screen_height": 480,

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
    "duration": 40,  # [s]
    "ego_spacing": 2,
    "vehicles_density": 1,


    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,  # The reward received at each lane change action.

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
    PPOConfig()
    .environment(env = ENV_ID,
                 env_config= ENV_CONFIG
    )
    .framework("torch")

    .env_runners(
        num_env_runners=4,  #engaging 4 gpu cores
        num_envs_per_env_runner=2, #each core simulating 2 envs

        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
        sample_timeout_s=180.0,
        rollout_fragment_length=400 
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=0
        )
    .training(
        train_batch_size = 6400,
        lr=5e-5 
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )

    .callbacks(  ##TRY REMOVING
        MyCallbacks
    )

)


algo = config.build_algo()
log_dir = algo.logdir

# Training
for i in range(10):
    result = algo.train()
    
    reward = result['env_runners']['episode_return_mean'] #TRY REMOVING
    pbar.set_postfix({
        "Reward": f"{reward:.2f}",
        "Batch": result['train_batch_size'],
    })

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
    print(f"\n@@@ I risultati di addestramento sono salvati in: {train_files[0]} \n")
    print(f"execute: tensorboard --logdir={train_files[0]} to see results") 
else:
    print("\n$$$ Attenzione: il file result.json non è stato trovato.")