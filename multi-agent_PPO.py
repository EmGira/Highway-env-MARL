#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html#ray.tune.CheckpointConfig
#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html#ray.tune.RunConfig
#https://github.com/ray-project/ray/issues/51560#issuecomment-2758195710 thread for AdamBetas Fix

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback, FixAdamBetasCallback
from configs.intersection.IntersectionConfigs import get_default_multi_agent_config, get_busy_intersection_config
import highway_env

import ray
from ray import shutdown
from ray import tune
from ray.tune import RunConfig, CheckpointConfig, FailureConfig

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import ASHAScheduler


import os
from pathlib import Path
import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if ray.is_initialized():
    ray.shutdown()
ray.init(object_store_memory=800 * 1024 * 1024)


today = datetime.date.today()

if not os.path.isdir(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}"):
    os.mkdir(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}")

checkpoints_dir = Path(f"./A-checkpoints/{today.strftime('%Y-%m-%d')}")
nr_of_subdirectories = len([f for f in checkpoints_dir.iterdir() if f.is_dir()])



#CONFIG
NR_AGENTS = 2
ENV_CONFIG = get_busy_intersection_config(NR_AGENTS)
#ENV_CONFIG["duration"] = 200


tune.register_env("intersection-v1_multiagent", lambda config: RLlibHighwayWrapper(ENV_CONFIG, "intersection-v1")) #TODOO.1 add envID as parameter

config = (
    PPOConfig()
    .environment(
        env = "intersection-v1_multiagent"
    )
    .framework("torch")
    .env_runners(
        num_env_runners=6,  #engaging 4 gpu cores
        num_envs_per_env_runner=1, #each core simulating 1 envs

        #env_to_module_connector=lambda env, spaces, device: FlattenObservations(), this caused issues, flattening happens in the MA_wrapper now
        sample_timeout_s=200.0,
        rollout_fragment_length="auto",  #nr of steps each env runner takes before sending to learner, ( total_train_batch_size / (num_env_runners * num_env_per_env_runner) )

       

        
    )
    .evaluation(
        evaluation_num_env_runners=0,
        evaluation_interval=5,
        evaluation_duration=20,
        evaluation_duration_unit="episodes",
        
    
   
    )
    .training(
        #GENERAL RL configs
        train_batch_size_per_learner = tune.grid_search([2048]),  #total train_batch_size = 2048 * 1 learner
        minibatch_size = 128,
        num_epochs = 5,
        
   
        
        lr = tune.grid_search([1e-4]), # 1e-4
        gamma = 0.99,

        #PPO specific configs

        use_critic = True,
        use_gae= True,
        lambda_= 1, #default = 1

    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )
    .multi_agent(
        policies={"agent_0", "agent_1"}, 
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id, 
    )
    .callbacks([CrashLoggerCallback, FixAdamBetasCallback] )

)


run_config = RunConfig(

    name=f"RunID_{nr_of_subdirectories}",
    storage_path=os.path.abspath(checkpoints_dir),
    
    stop={"training_iteration": 500},

    failure_config=FailureConfig(
        max_failures=0,
    ),

    callbacks=None,

   
    checkpoint_config=CheckpointConfig(
        num_to_keep = 3,  # keep the 3 most recent (or best) checkpoints from training
        checkpoint_score_attribute = "env_runners/episode_return_mean", # The attribute that will be used to score checkpoints to determine which of the *num_to_keep* checkpoints should be kept
        checkpoint_score_order = 'max',
        checkpoint_frequency=10, #save a checkpoint every 10 iterations
        checkpoint_at_end=True  #always save checkpoint at end of training
         

    )
)


def custom_trial_dirname(trial):

    batch_size = trial.config.get("_train_batch_size_per_learner")
    lr = trial.config.get("lr")
    return f"PPO_Batch_{batch_size}-lr_{lr}_ID_{trial.trial_id}"

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        num_samples=1,
        # scheduler=ASHAScheduler(metric="env_runners/episode_return_mean", mode= "max", grace_period=50, max_t=run_config.stop["training_iteration"]),
        trial_dirname_creator=custom_trial_dirname,
        trial_name_creator=lambda trial: f"Experiment_{trial.trial_id}"
    ),            
    param_space=config,         
    run_config=run_config,    
)


# tuner = tune.Tuner.restore(   #to restore from checkpoint
#     path=os.path.abspath(""), 
#     trainable="PPO",
#     resume_unfinished=True,
#     resume_errored = True,

# )


#TRAIN
print("\n@@@ Initializing training...")
results = tuner.fit()



#EXTRACT
best_result = results.get_best_result(
    metric = run_config.checkpoint_config.checkpoint_score_attribute, 
    mode = run_config.checkpoint_config.checkpoint_score_order
)
print("\n@@@ Training completed!")
print(f"\t Results store here ==> {best_result.path}")
print(f"\t For TensoBoard, execute ==> tensorboard --logdir={best_result.path}")

shutdown()