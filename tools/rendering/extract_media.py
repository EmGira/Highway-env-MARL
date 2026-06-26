import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import torch
import numpy as np
import ray
from ray.rllib.models import ModelCatalog
import os
import sys
import imageio
import shutil

parent_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_folder)

from utils.evaluation_utils import load_policy_or_module, create_eval_env, compute_actions, compute_actions_legacy
from configs.intersection.IntersectionConfigs import get_ego_only_config

from src.models.CentralizedCriticModel import CentralizedCriticModel
from src.models.CentralizedCriticSACModel import CentralizedCriticSACModel

ModelCatalog.register_custom_model("centralized_critic_model", CentralizedCriticModel)
ModelCatalog.register_custom_model("centralized_critic_sac_model", CentralizedCriticSACModel)

def extract_media(checkpoint_path, env_name, num_agents, target_outcome, output_prefix, max_episodes=50):
    print(f"Extracting {output_prefix}...")
    
    ENV_CONFIG = get_ego_only_config(num_agents=num_agents)
    ENV_CONFIG["simulation_frequency"] = 15
    ENV_CONFIG["randomize_controlled_vehicles"] = False
    
    try:
        model_or_policy, stack_type = load_policy_or_module(checkpoint_path)
    except Exception as e:
        print(f"Failed to load policy from {checkpoint_path}: {e}")
        return
        
  
    search_env = create_eval_env(stack_type, ENV_CONFIG, env_name, render_mode=None, inference_mode=True)
    
    found_seed = None
    
    for ep in range(max_episodes):
        seed = ep + 100 
        obs, info = search_env.reset(seed=seed)
        done = {"__all__": False}
        truncated = {"__all__": False}
        
        while not (done["__all__"] or truncated["__all__"]):
            if stack_type == "new":
                agents_actions = compute_actions(model_or_policy, obs)
            else:
                agents_actions = compute_actions_legacy(model_or_policy, obs)
            
            obs, reward, done, truncated, info = search_env.step(agents_actions)
        
        last_info = list(info.values())[0] if info else {}
        is_crashed = last_info.get('crashed', False)
        is_success = last_info.get("all_arrived", False)
        
        outcome = "truncated"
        if is_success:
            outcome = "success"
        elif is_crashed:
            outcome = "crashed"
            
        print(f"  Search Ep {ep} (seed={seed}) outcome: {outcome}")
        
        if target_outcome == "any" or outcome == target_outcome:
            found_seed = seed
            print(f"  Target outcome '{target_outcome}' found at seed {seed}!")
            break
            
    search_env.close()
    
    if found_seed is None:
        print(f"  Failed to find an episode with outcome '{target_outcome}' after {max_episodes} attempts.")
        return

   
    print(f"  Re-running seed {found_seed} with rendering to capture media...")
    RENDER_MODE = "rgb_array"
    ma_env = create_eval_env(stack_type, ENV_CONFIG, env_name, render_mode=RENDER_MODE, inference_mode=True)
    
    temp_video_dir = output_prefix + "_temp_video"
    if os.path.exists(temp_video_dir):
        shutil.rmtree(temp_video_dir)
        
   
    ma_env.env.env.metadata["render_fps"] = 30
    
    video_env = RecordVideo(ma_env.env.env, video_folder=temp_video_dir, episode_trigger=lambda e: True)
    ma_env.env.env = video_env
    
    video_env.unwrapped.set_record_video_wrapper(video_env)
    
    obs, info = ma_env.reset(seed=found_seed)
    
   
    viewer = getattr(ma_env.env.env.unwrapped, 'viewer', None)
    if viewer is not None:
        viewer.window_position = lambda: np.array([0, 0])
  

    done = {"__all__": False}
    truncated = {"__all__": False}
    
    while not (done["__all__"] or truncated["__all__"]):
        if stack_type == "new":
            agents_actions = compute_actions(model_or_policy, obs)
        else:
            agents_actions = compute_actions_legacy(model_or_policy, obs)
        
        obs, reward, done, truncated, info = ma_env.step(agents_actions)
    
    ma_env.close()
    
 
    for file in os.listdir(temp_video_dir):
        if file.endswith(".mp4"):
            mp4_path = os.path.join(temp_video_dir, file)
            mp4_filename = f"{output_prefix}.mp4"
            shutil.copy(mp4_path, mp4_filename)
            print(f"  Successfully saved MP4: {mp4_filename}")
            break
    

    shutil.rmtree(temp_video_dir)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    IPPO_CKPT = os.path.abspath("./A-checkpoints/TEST/3agents-IPPO/PPO_0/ID_4cd2d_00000/checkpoint_000005")
    MAPPO_CKPT = os.path.abspath("./A-checkpoints/TEST/3agents-MAPPO/MAPPO_2/ID_a2e6e_00000/checkpoint_000005")
    MASAC_CKPT = os.path.abspath("./A-checkpoints/3agents-MASAC-Failure/MASAC_0/ID_bb8e8_00000/checkpoint_000003")
    
    OUTPUT_DIR = os.path.abspath("./thesis_media")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
  
    # extract_media(IPPO_CKPT, "customIntersection-env-v0", 6, "success", f"{OUTPUT_DIR}/IPPO_Intersection_5agents_Success")
    # extract_media(IPPO_CKPT, "customIntersection-env-v0", 6, "crashed", f"{OUTPUT_DIR}/IPPO_Intersection_5agents_Crash")
    

    # extract_media(MAPPO_CKPT, "customIntersection-env-v0", 6, "success", f"{OUTPUT_DIR}/MAPPO_Intersection_5agents_Success")
    # extract_media(MAPPO_CKPT, "customIntersection-env-v0", 6, "crashed", f"{OUTPUT_DIR}/MAPPO_Intersection_5agents_Crash")
    
  
    
    # extract_media(MAPPO_CKPT, "customRoundabout-env-v0", 6, "success", f"{OUTPUT_DIR}/MAPPO_Roundabout_6agents_Success")
    # extract_media(MAPPO_CKPT, "customRoundabout-env-v0", 6, "crashed", f"{OUTPUT_DIR}/MAPPO_Roundabout_6agents_Crash")

    # extract_media(IPPO_CKPT, "customRoundabout-env-v0", 6, "success", f"{OUTPUT_DIR}/IPPO_Roundabout_6agents_Success")
    # extract_media(IPPO_CKPT, "customRoundabout-env-v0", 6, "crashed", f"{OUTPUT_DIR}/IPPO_Roundabout_6agents_Crash")


    # extract_media(MASAC_CKPT, "customIntersection-env-v0", 3, "truncated", f"{OUTPUT_DIR}/MASAC_Intersection_3agents_Freezing")
    
    ray.shutdown()
    
