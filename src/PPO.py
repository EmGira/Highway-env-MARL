#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html#ray.tune.CheckpointConfig
#https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html#ray.tune.RunConfig
#https://github.com/ray-project/ray/issues/51560#issuecomment-2758195710 thread for AdamBetas Fix

import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)

from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
from utils.callbacks.Callbacks import CrashLoggerCallback, FixAdamBetasCallback, SafeEvaluationCallback
from configs.intersection.IntersectionConfigs import get_simple_multi_agent_config, get_improved_Simple_config
import highway_env

import ray
from ray import shutdown
from ray import tune
from ray.tune import RunConfig, CheckpointConfig, FailureConfig

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from optuna.storages import RDBStorage

from pathlib import Path
import datetime


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

nr_of_subdirectories, checkpoints_dir, today = initialize()


#CONFIG
NR_AGENTS = 2
ENV_CONFIG = get_improved_Simple_config(NR_AGENTS)

ENV_CONFIG["spawn_points"] = ["3", "1"]
ENV_CONFIG["multi_destinations"] = ["o0", "o3"]

# ENV_CONFIG["spawn_points"] = ["0", "1"]
# ENV_CONFIG["multi_destinations"] = ["o1", "o0"]


tune.register_env("CustomIntersection-env-v0", lambda config: RLlibHighwayWrapper(ENV_CONFIG, "customIntersection-env-v0")) #

config = (
    PPOConfig()
    .environment(
        env = "CustomIntersection-env-v0"
    )
    .framework("torch")
    .env_runners(
        num_env_runners=6,  
        num_envs_per_env_runner=1,
        sample_timeout_s=200.0,
        rollout_fragment_length="auto",  #nr of steps each env runner takes before sending to learner, ( total_train_batch_size / (num_env_runners * num_env_per_env_runner) )
    )
    .evaluation(
        evaluation_num_env_runners=0,
        evaluation_interval=10,
        evaluation_duration=30,
        evaluation_duration_unit="episodes",
        
        

    )
    .training( 
        
        train_batch_size_per_learner=4096,
        minibatch_size=256,          
        clip_param=0.1,                 
        
      
        entropy_coeff = 0.0028,
        num_epochs = 4,
        lr = [[0, 1e-4], [1000000, 1e-6]],

        gamma = 0.95, #before: 0.995


        use_critic = True,           
        use_gae = True,               

        lambda_ = 0.95,
        vf_loss_coeff = 1,    # 0.5


        kl_target = 0.01,     

        #policy and value function dont share weights
        model={
        "vf_share_layers": False,
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
    .callbacks([CrashLoggerCallback, FixAdamBetasCallback, SafeEvaluationCallback] )

)

run_config = RunConfig(

    name=f"PPO_{nr_of_subdirectories}",
    storage_path=os.path.abspath(checkpoints_dir),
    
    stop={"training_iteration": 100},


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


optuna_storage = RDBStorage(url="sqlite:///optuna_highway_results.db")
study_name = f"PPO_Study_{today.strftime('%Y-%m-%d')}_Run_{nr_of_subdirectories}"
algo = OptunaSearch(
    storage=optuna_storage,
    study_name=study_name,

)

scheduler = ASHAScheduler(    
    max_t=run_config.stop["training_iteration"],                    
    grace_period=30, 
    reduction_factor=2
)



def custom_trial_dirname(trial):
    
    lr_config = trial.config.get("lr")

    if not isinstance(lr_config, list):
        lr_str = f"{lr_config:.7f}"
    else:
        lr_str = "scheduled"
        
    return f"lr_{lr_str}_ID_{trial.trial_id}"
    



def custom_trial_name(trial):
    return f"Experiment_{trial.trial_id}"

tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(

        metric=run_config.checkpoint_config.checkpoint_score_attribute, 
        mode=run_config.checkpoint_config.checkpoint_score_order,

        num_samples=1,

        #search_alg=algo,
        #scheduler=scheduler, 

        trial_dirname_creator=custom_trial_dirname,
        trial_name_creator=custom_trial_name
    ),            
    param_space=config,         
    run_config=run_config,    
)


# tuner = tune.Tuner.restore(   
#     path=os.path.abspath("./A-checkpoints/2026-04-29/PPO_3"), 
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





