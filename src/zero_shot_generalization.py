    # config B = 
    #     ["spawn_points"] = ["3", "1"]
    #     ["multi_destinations"] = ["o0", "o3"]
    
    # config A (default) = 
    #     ["spawn_points"] = ["0", "1"]
    #     ["multi_destinations"] = ["o1", "o0"]

    # config C =
    #      ["spawn_points"] = ["2", "1"] 
    #      ["multi_destinations"] = ["o3", "o3"] 

import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)


import ray
import numpy as np
import matplotlib.pyplot as plt
import torch

from configs.intersection.IntersectionConfigs import get_simple_multi_agent_config, get_improved_Simple_config

ray.init(ignore_reinit_error=True)


def compute_actions(multi_rl_module, obs):

    policy_module = multi_rl_module["shared_policy"]

    with torch.no_grad():
        agents_actions = {}
        for agent_id, agent_obs in obs.items():
       
            ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
            output = policy_module.forward_inference({"obs": ao})
            agents_actions[agent_id] = torch.argmax(output["action_dist_inputs"], dim=1).item()

    return agents_actions

def compute_continous_actions(multi_rl_module, obs, env_agent_ids):
    policy_module = multi_rl_module["shared_policy"]

    with torch.no_grad():
        agents_actions = {}
        
        for agent_id in env_agent_ids:
            
            if agent_id in obs:
   
                agent_obs = obs[agent_id]
                ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
                output = policy_module.forward_inference({"obs": ao})
                
                action_dist_params = output["action_dist_inputs"][0].cpu().numpy()
                
                greedy_action = np.clip(
                    action_dist_params[0:1], 
                    a_min=-1.0,
                    a_max=1.0,
                )
                agents_actions[agent_id] = greedy_action
                
            else:
                agents_actions[agent_id] = np.array([0.0], dtype=np.float32)
                
    return agents_actions


@ray.remote(num_cpus=1)
def distributed_evaluate_worker(checkpoint_path, env_config, num_episodes):
    import torch
    from ray.rllib.core.rl_module import MultiRLModule
    from pathlib import Path
    from utils.wrapper.MA_wrapper import RLlibHighwayWrapper
 
    multi_rl_module = MultiRLModule.from_checkpoint(
        Path(checkpoint_path) / "learner_group" / "learner" / "rl_module"
    )
    env = RLlibHighwayWrapper(config=env_config, env_id="customIntersection-env-v0", render_mode=None)

    worker_history = {
        "rewards": [],
        "crashes": [],   
        "successes": [] 
    }

    for ep in range(num_episodes):
        obs, info = env.reset()
        all_agent_ids = list(obs.keys())
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        ep_reward = 0

        while not (terminated["__all__"] or truncated["__all__"]):
            
            agents_actions = compute_actions(multi_rl_module, obs)

            obs, reward, terminated, truncated, info = env.step(agents_actions)
            ep_reward += sum(reward.values()) 

        last_info = list(info.values())[0] if info else {}
        worker_history["rewards"].append(ep_reward)
        worker_history["crashes"].append(1 if last_info.get('crashed', False) else 0)
        worker_history["successes"].append(1 if last_info.get('all_arrived', False) else 0)
        
    env.close()
    return worker_history


def run_distributed_evaluation(policy_name, checkpoint_path, env_config, total_episodes=100, num_workers=4):
    print(f"\n{'='*50}")
    print(f"Validating: {policy_name} on {num_workers} workers")
    print(f"{'='*50}")
    

    episodes_per_worker = total_episodes // num_workers
    
    
    futures = [
        distributed_evaluate_worker.remote(checkpoint_path, env_config, episodes_per_worker) 
        for _ in range(num_workers)
    ]
    
    
    results = ray.get(futures)
    
    
    aggregated_history = {"rewards": [], "crashes": [], "successes": []}
    for res in results:
        aggregated_history["rewards"].extend(res["rewards"])
        aggregated_history["crashes"].extend(res["crashes"])
        aggregated_history["successes"].extend(res["successes"])
        
    
    rewards_array = np.array(aggregated_history["rewards"])
    mean_reward = np.mean(rewards_array)
    std_reward = np.std(rewards_array) 
    
    print(f"Results for {policy_name}:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return aggregated_history, mean_reward, std_reward


NR_AGENTS = 2
NUM_TEST_EPISODES = 200
NUM_WORKERS = 7 

def get_base_config():
    config = get_improved_Simple_config(num_agents=NR_AGENTS)
    config["simulation_frequency"] = 15
    return config


checkpoint = "./A-checkpoints/run12/PPO_0/lr_scheduled_ID_a12e7_00000/checkpoint_000021"
scenarios = [
    {
        "name": "31-o0o3 on 31-o0o3 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["3", "1"], "multi_destinations": ["o0", "o3"]} #NATIVE, the policy evaluated on the config it was trained on
    },
    #SPECULAR, to test relative observations
    {
        "name": "31-o0o3 on 13-o3o0 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["1", "3"], "multi_destinations": ["o3", "o0"]}
    },
    {
        "name": "31-o0o3 on 20-o3o2 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["2", "0"], "multi_destinations": ["o3", "o2"]}
    },

    # "HARD maps" new scenarios where the agents cross paths 
    {   
        "name": "31-o0o3 on 21-o3o3 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["2", "1"], "multi_destinations": ["o3", "o3"]}
    },
    {   
        "name": "31-o0o3 on 01-o2o3 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["0", "1"], "multi_destinations": ["o2", "o3"]}
    },
    {   
        "name": "31-o0o3 on 12-o3o0 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["1", "2"], "multi_destinations": ["o3", "o0"]}
    },
    {   
        "name": "31-o0o3 on 11-o2o2 ",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["1", "1"], "multi_destinations": ["o2", "o2"]}
    },

    # "EASY maps" new scenarios where the agents do not cross paths:
    {
        "name": "31-o0o3 on 01-o1o0 ",                          
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["0", "1"], "multi_destinations": ["o1", "o0"]} 
    },
    {
        "name": "31-o0o3 on 02-o0o2",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["0", "2"], "multi_destinations": ["o0", "o2"]}
    },
    {
        "name": "31-o0o3 on 31-o1o3",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "spawn_points": ["3", "1"], "multi_destinations": ["o1", "o3"]}
    },
 
]

results = []
for scenario in scenarios:
    history, mean, std = run_distributed_evaluation(
        scenario["name"], scenario["checkpoint"], scenario["config"], NUM_TEST_EPISODES, NUM_WORKERS
    )
    results.append({
        "name": scenario["name"],
        "history": history,
        "mean": mean,
        "std": std
    })


import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(results):
    if not results:
        return
        
    #sort by success rate
    results_sorted = sorted(
        results, 
        key=lambda x: sum(x["history"]["successes"]) / len(x["history"]["successes"]), 
        reverse=True
    )
    
    num_episodes = len(results_sorted[0]["history"]["crashes"])
    
 
    labels = [res["name"].split(" on ")[-1] if " on " in res["name"] else res["name"] for res in results_sorted]
    
    rewards_data = [res["history"]["rewards"] for res in results_sorted]

    success_rates = [(sum(res["history"]["successes"]) / num_episodes) * 100 for res in results_sorted]
    crash_rates = [(sum(res["history"]["crashes"]) / num_episodes) * 100 for res in results_sorted]
   
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle('Zero-Shot Generalization Comparison', fontsize=18, fontweight='bold')



    bplot = axs[0].boxplot(rewards_data, labels=labels, patch_artist=True, 
                           medianprops=dict(color="black", linewidth=1.5))
    

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(labels)))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axs[0].set_title('Reward Distribution per Configuration', fontsize=14)
    axs[0].set_ylabel('Episode Reward', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.4, axis='y')
    # Ruotiamo le etichette per non farle accavallare
    axs[0].tick_params(axis='x', rotation=45) 

    

    x = np.arange(len(labels))  
    width = 0.35 

   
    rects1 = axs[1].bar(x - width/2, success_rates, width, label='Success Rate', color='forestgreen', alpha=0.8)
    rects2 = axs[1].bar(x + width/2, crash_rates, width, label='Crash Rate', color='crimson', alpha=0.8)

    axs[1].set_title('Final Evaluation Metrics (200 Episodes)', fontsize=14)
    axs[1].set_ylabel('Percentage (%)', fontsize=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=45, ha='right') 
    axs[1].set_ylim([0, 105])
    axs[1].grid(True, linestyle='--', alpha=0.4, axis='y')
    axs[1].legend(loc='upper right', fontsize=12)

    
    axs[1].bar_label(rects1, fmt='%.0f%%', padding=3, fontsize=9)
    axs[1].bar_label(rects2, fmt='%.0f%%', padding=3, fontsize=9)

    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) 
    
    plt.savefig("ZSG_run5.svg", format="svg")
    plt.show()


plot_comparison(results)

ray.shutdown()
