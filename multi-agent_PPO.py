#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html#ray.tune.CheckpointConfig
#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html#ray.tune.RunConfig
#https://github.com/ray-project/ray/issues/51560#issuecomment-2758195710 thread for AdamBetas Fix

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback, FixAdamBetasCallback
from configs.intersection.IntersectionConfigs import get_default_multi_agent_config, get_busy_intersection_config, get_simple_multi_agent_config
from configs.CustomMerge.customMergeConfigs import get_default_custom_env_config
import highway_env

import ray
from ray import shutdown
from ray import tune
from ray.tune import CLIReporter, RunConfig, CheckpointConfig, FailureConfig

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


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

def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return agent_id


def calculate_minibatch_size(config):

    batch_size = config.get("train_batch_size_per_learner", config.get("_train_batch_size_per_learner"))
    if batch_size == 1024:
        return 64
    return 128



#CONFIG
NR_AGENTS = 2
ENV_CONFIG = get_busy_intersection_config(NR_AGENTS)



# tune.register_env("intersection-v1_multiagent", lambda config: RLlibHighwayWrapper(ENV_CONFIG, "intersection-v1")) #
tune.register_env("CustomIntersection-env-v0", lambda config: RLlibHighwayWrapper(ENV_CONFIG, "customIntersection-env-v0")) #

config = (
    PPOConfig()
    .environment(
        env = "CustomIntersection-env-v0"
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
        train_batch_size_per_learner=tune.choice([1024, 2048]),
        
        minibatch_size=tune.sample_from(calculate_minibatch_size),
        num_epochs = tune.randint(5, 30),
   
        
        lr = tune.loguniform(1e-5, 5e-4), # 1e-4
        gamma = tune.choice([0.95, 0.99, 0.995, 0.999]),

        #PPO specific configs
        use_critic = True,
        use_gae= True,
        lambda_= 1, #default = 1
        entropy_coeff = tune.loguniform(1e-4, 1e-2),
        clip_param = tune.choice([0.1, 0.2, 0.3])

      

    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=1
    )
    .multi_agent(
        policies={"agent_0", "agent_1"}, 
        policy_mapping_fn= my_policy_mapping_fn, 
    )
    .callbacks([CrashLoggerCallback, FixAdamBetasCallback] )

)





run_config = RunConfig(

    name=f"RunID_{nr_of_subdirectories}",
    storage_path=os.path.abspath(checkpoints_dir),
    
    stop={"training_iteration": 5},


    failure_config=FailureConfig(
        max_failures=0,
    ),


    callbacks=None,

   
    checkpoint_config=CheckpointConfig(
        num_to_keep = 3,  # keep the 3 most recent (or best) checkpoints from training
        checkpoint_score_attribute = "env_runners/episode_return_mean", # The attribute that will be used to score checkpoints to determine which of the *num_to_keep* checkpoints should be kept
        checkpoint_score_order = 'max',
        checkpoint_frequency=2, #save a checkpoint every 10 iterations
        checkpoint_at_end=True  #always save checkpoint at end of training      

    )

)

algo = OptunaSearch()
# scheduler = ASHAScheduler(    
#     max_t=run_config.stop["training_iteration"],                    
#     grace_period=50, # tries at least 50 iters before stopping
#     reduction_factor=2
# )



def custom_trial_dirname(trial):

    batch_size = trial.config.get("_train_batch_size_per_learner")
    lr = trial.config.get("lr")
    return f"PPO_Batch_{batch_size}-lr_{lr}_ID_{trial.trial_id}"
def custom_trial_name(trial):
    return f"Experiment_{trial.trial_id}"

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(

        metric=run_config.checkpoint_config.checkpoint_score_attribute, 
        mode=run_config.checkpoint_config.checkpoint_score_order,

        num_samples=10,

        search_alg=algo,
        #scheduler=scheduler, TODOO scheduler disabled for now as it is not behaving as expected

        trial_dirname_creator=custom_trial_dirname,
        trial_name_creator=custom_trial_name
    ),            
    param_space=config,         
    run_config=run_config,    
)


# tuner = tune.Tuner.restore(   #to restore from checkpoint
#     path=os.path.abspath("./A-checkpoints/2026-04-02/RunID_5"), 
#     trainable="PPO",
#     resume_unfinished=True,
#     resume_errored = True,
#     param_space=config, 

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



