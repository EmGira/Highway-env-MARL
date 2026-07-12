#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html#ray.tune.CheckpointConfig
#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html#ray.tune.RunConfig
#https://github.com/ray-project/ray/issues/51560#issuecomment-2758195710 thread for AdamBetas Fix

import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)

from src.models.CentralizedCriticModel import CentralizedCriticModel
from utils.wrapper.MAPPO_wrapper import RLlibMAPPOHighwayWrapper
from utils.callbacks.MAPPO_callbacks import MAPPOCrashLoggerCallback, MAPPOFixAdamBetasCallback, MAPPOSafeEvaluationCallback
from configs.intersection.IntersectionConfigs import get_ego_only_config
import highway_env



import ray
from ray import shutdown
from ray import tune
from ray.tune import RunConfig, CheckpointConfig, FailureConfig
from ray.rllib.models import ModelCatalog

from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from optuna.storages import RDBStorage


from pathlib import Path
import datetime
import argparse



tune.register_trainable("MAPPO", PPO)

import gymnasium as gym

ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)


def initialize():
        
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if ray.is_initialized():
        ray.shutdown()
    ray.init(object_store_memory=800 * 1024 * 1024)


    today = datetime.date.today()
    if not os.path.isdir(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}"):
        os.mkdir(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}")

    checkpoints_dir = Path(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}")
    nr_of_subdirectories = len([f for f in checkpoints_dir.iterdir() if f.is_dir()])

    return nr_of_subdirectories, checkpoints_dir, today

def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return agent_id

def custom_trial_dirname(trial):
    return f"ID_{trial.trial_id}"

def custom_trial_name(trial):
    return f"Experiment_{trial.trial_id}"




if __name__ == "__main__":

  
    parser = argparse.ArgumentParser(description="Script to start or resume training with Ray Tune")
    parser.add_argument("--resume", type=str, default=None, help="Path to the Run Folder from wich we want to resume training")
    parser.add_argument("--enable_scheduler", action="store_true", help="Enables the ASHA scheduler for early stopping")
    parser.add_argument("--enable_optuna", action="store_true", help="Enables Optuna for HyperParam search")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--iterations", type=int, default=200, help="Number of training iterations")
    args = parser.parse_args()


    nr_of_subdirectories, checkpoints_dir, today = initialize()

    ENV_CONFIG = get_ego_only_config(3)
    ENV_CONFIG["randomize_controlled_vehicles"] = False
    
    tune.register_env("CustomIntersection-env-v0", lambda config: RLlibMAPPOHighwayWrapper(config, "customIntersection-env-v0")) #


    config = (
        PPOConfig()
        .environment(
            env = "CustomIntersection-env-v0",
            env_config = ENV_CONFIG
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=6,  
            num_envs_per_env_runner=1,
            sample_timeout_s=200.0,
            rollout_fragment_length="auto",  #nr of steps each env runner takes before sending to learner, ( total_train_batch_size / (num_env_runners * num_env_per_env_runner) )
        )
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=10,
            evaluation_duration=60,
            evaluation_duration_unit="episodes", 

        )
        .training( 
            
            train_batch_size_per_learner=16384,
            minibatch_size=2048,          
            clip_param=0.2,                 
            
        
            entropy_coeff = 0.01,
            num_epochs = 5,
            
            #lr = [[0, 3e-4], [10000000, 1e-5]],
            lr=3e-4, #3e-4
            
            gamma = 0.975, #before: 0.95


            use_critic = True,           
            use_gae = True,               
            lambda_ = 0.95,
            vf_loss_coeff = 0.5,    # 0.5
            kl_target = 0.02,     

            # Use the registered custom model for MAPPO
            model={
                "custom_model": "centralized_critic_model",
                "custom_model_config": {
                    "fcnet_hiddens": [256, 256],
                }
            }  
            
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1
        )
        .multi_agent(

            policies={"shared_policy"}, 
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .callbacks(make_multi_callbacks([MAPPOCrashLoggerCallback, MAPPOFixAdamBetasCallback, MAPPOSafeEvaluationCallback]))

    )

    if args.seed is not None:
        config = config.debugging(seed=args.seed)

    run_name = f"MAPPO_seed_{args.seed}" if args.seed is not None else f"MAPPO_{nr_of_subdirectories}"
    run_config = RunConfig(

        name=run_name,

        storage_path=os.path.abspath(checkpoints_dir),
        
        stop={"training_iteration": args.iterations},

        
        failure_config=FailureConfig(
            max_failures=0,
        ),
    
        checkpoint_config=CheckpointConfig(
            num_to_keep = 3,
            checkpoint_score_attribute = "safe_return_mean",
            checkpoint_score_order = 'max',
            checkpoint_frequency=10, 
            checkpoint_at_end=True 
        )

    )


    algo = None
    scheduler = None
    if args.enable_optuna:
        optuna_storage = RDBStorage(url="sqlite:///optuna_highway_results.db")
        study_name = f"MAPPO_Study_{today.strftime('%Y-%m-%d')}_Run_{nr_of_subdirectories}"
        algo = OptunaSearch(
            storage=optuna_storage,
            study_name=study_name,
        )
    if args.enable_scheduler:
        scheduler = ASHAScheduler(    
            max_t=run_config.stop["training_iteration"],                    
            grace_period=30, 
            reduction_factor=2
        )


    if args.resume is not None:
        print(f"\n@@@ Resuming training from {args.resume}...")
        tuner = tune.Tuner.restore(   
            path=os.path.abspath(args.resume), 
            trainable="MAPPO",
            resume_unfinished=True,
            resume_errored=True,
            
        )
    else:
        print("\n@@@ Initializing NEW training...")
        tuner = tune.Tuner(
            "MAPPO",
            tune_config=tune.TuneConfig(
                metric=run_config.checkpoint_config.checkpoint_score_attribute, 
                mode=run_config.checkpoint_config.checkpoint_score_order,
                num_samples=1,
                trial_dirname_creator=custom_trial_dirname,
                trial_name_creator=custom_trial_name,
                search_alg=algo,
                scheduler=scheduler, 
            ),            
            param_space=config,        
            run_config=run_config,    
        )


    results = tuner.fit()


    print("\n@@@ Training completed!")


    shutdown()
