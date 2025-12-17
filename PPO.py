import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray import tune
import gymnasium as gym
import highway_env
import os
import glob
from pathlib import Path
import datetime


today = datetime.date.today()
checkpoints_dir = Path("./checkpoints")
nr_of_subdirectories = len([f for f in checkpoints_dir.iterdir() if f.is_dir()])

ray.init(log_to_driver=False, configure_logging=True, logging_level="ERROR")


ENV_ID = 'highway-v0'
HIGHWAY_ENV_CONFIG = {
  "lanes_count": 2,
    "observation": {
        "type": "Kinematics"
    },
    "duration": 20
}

def env_creator(env_config):
    import highway_env
    return gym.make(ENV_ID, render_mode="rgb_array", config=env_config)

tune.register_env(ENV_ID, env_creator)

# Configure the algorithm.
config = (
    PPOConfig()
    .environment(env = ENV_ID,
                 env_config= HIGHWAY_ENV_CONFIG
    )
    .framework("torch")
    .env_runners(
        num_env_runners=2,

        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
        sample_timeout_s=180.0,
        rollout_fragment_length=100
    )
    .evaluation(evaluation_num_env_runners=1)
    .training(
        train_batch_size = 200
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )
)


# Build the algorithm.
algo = config.build_algo()
log_dir = algo.logdir

# Training
for i in range(2):
    result = algo.train()
    print(f"Iterazione {i + 1}: Ricompensa Media (Train): {result['env_runners']['episode_return_mean']:.2f}")


algo.evaluate()


train_files = glob.glob(os.path.join(log_dir, "result.json"))


path = f"./checkpoints/{today.strftime('%Y-%m-%d')}_ID_{nr_of_subdirectories}"
saved_results = algo.save(checkpoint_dir = os.path.abspath(path))
checkpoint_dir = saved_results.checkpoint.path


if checkpoint_dir:
    print(f"\n@@@ Checkpoint salvato correttamente!")
    print(f"@@@ Percorso: {checkpoint_dir}\n")

# Release the algo instance's resources (EnvRunners and Learners).
algo.stop()


if train_files:
    print(f"\n@@@ I risultati di addestramento sono salvati in: {train_files[0]} \n")
    print(f"execute: tensorboard --logdir={train_files[0]} to see results") 
else:
    print("\n$$$ Attenzione: il file result.json non è stato trovato.")