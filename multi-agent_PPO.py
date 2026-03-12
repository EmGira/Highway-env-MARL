#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html#ray.tune.CheckpointConfig
#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html#ray.tune.RunConfig
#https://github.com/ray-project/ray/issues/51560#issuecomment-2758195710 thread for AdamBetas Fix

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback, FixAdamBetasCallback
from configs.intersection.IntersectionConfigs import get_multi_agent_config
import highway_env

from ray import tune
from ray.tune import RunConfig, CheckpointConfig, FailureConfig

from ray.rllib.algorithms.ppo import PPOConfig


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
    PPOConfig()
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
        
        # evaluation_config={
        #     "env": "intersection-v1_multiagent",
        #     "explore": False, 
        # }
   
        )
    .training(
    
        train_batch_size_per_learner=4000, #total train_batch_size = 4000 * 1 learner
        minibatch_size = 256,
        num_epochs = 10,
        
     
        #lr=tune.grid_search([1e-4, 5e-5]) #this tells Tune to try both options with two trainings in parallel execution
        lr = 5e-5
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

    name=f"Run_{today.strftime('%Y-%m-%d')}_ID_{nr_of_subdirectories}",
    storage_path=os.path.abspath(checkpoints_dir),
    
    stop={"training_iteration": 500},

    failure_config=FailureConfig(
        max_failures=0,
    ),

    callbacks=None,

   
    checkpoint_config=CheckpointConfig(
        num_to_keep = 5,  # keep the 5 most recent checkpoints from training
        checkpoint_score_attribute = "env_runners/episode_return_mean", # The attribute that will be used to score checkpoints to determine which checkpoints should be kept
        checkpoint_score_order = 'max',
        checkpoint_frequency=10, #save a checkpoint every 10 iterations
        checkpoint_at_end=True  #always save checkpoint at end of training
         

    )
)


# tuner = tune.Tuner(
#     "PPO",                      
#     param_space=config,         
#     run_config=run_config       
# )


tuner = tune.Tuner.restore(   #to restore from checkpoint
    path=os.path.abspath("./A-checkpoints/2026-03-12/Run_2026-03-12_ID_1"), 
    trainable="PPO",
    resume_unfinished=True,
    resume_errored = True,

    param_space=config,

)


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