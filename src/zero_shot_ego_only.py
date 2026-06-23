import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)


import ray
import numpy as np
import matplotlib.pyplot as plt
import torch

from configs.intersection.IntersectionConfigs import get_improved_Simple_config, get_ego_only_config

ray.init(ignore_reinit_error=True)

from utils.evaluation_utils import load_policy_or_module, create_eval_env, compute_actions, compute_actions_legacy


@ray.remote(num_cpus=1)
def distributed_evaluate_worker(checkpoint_path, env_config, num_episodes):
    model_or_policy, stack_type = load_policy_or_module(checkpoint_path)
    env = create_eval_env(stack_type, env_config, ENV_ID, render_mode=None, inference_mode=False)

    worker_history = {
        "rewards": [],
        "crashes": [],   
        "successes": [] 
    }

    for ep in range(num_episodes):
        obs, info = env.reset()
        all_agent_ids = list(obs.keys())
        num_agents = len(all_agent_ids)
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        ep_reward = 0

        while not (terminated["__all__"] or truncated["__all__"]):
            if stack_type == "new":
                agents_actions = compute_actions(model_or_policy, obs)
            else:
                agents_actions = compute_actions_legacy(model_or_policy, obs, explore=False)

            obs, reward, terminated, truncated, info = env.step(agents_actions)
            ep_reward += sum(reward.values()) 

        normalized_ep_reward = ep_reward / num_agents if num_agents > 0 else 0.0
        worker_history["rewards"].append(normalized_ep_reward)

        last_info = list(info.values())[0] if info else {}
        worker_history["crashes"].append(1 if last_info.get('crashed', False) else 0)
        worker_history["successes"].append(1 if last_info.get('all_arrived', False) else 0)
        
    env.close()
    return worker_history


def run_distributed_evaluation(policy_name, checkpoint_path, env_config, total_episodes=100, num_workers=4):
    print(f"\n{'='*50}")
    print(f"Validating: {policy_name}")
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

def compute_duration(nAgents):
    if(nAgents <=    4):
        return 60
    else:
        return 60 + 10*nAgents
    


MAX_NR_AGENTS = 8
NUM_TEST_EPISODES = 200
NUM_WORKERS = 7 

ENV_ID = "customIntersection-env-v0"

def get_base_config():
    config = get_ego_only_config(3)
    config["simulation_frequency"] = 15
    config["randomize_controlled_vehicles"] = False
    return config


checkpoint = "./A-checkpoints/TEST/3agenti/PPO_0/ID_4cd2d_00000/checkpoint_000005"

scenarios = [
    {
        "name": f"{nAgents} agents",
        "checkpoint": os.path.abspath(checkpoint),
        "config": {**get_base_config(), "controlled_vehicles" : nAgents } 
    }   for nAgents in range(3, MAX_NR_AGENTS+1)
 
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
        
    
    results_sorted = sorted(
        results, 
        key=lambda x:  int(x["name"].split()[0]), #sum(x["history"]["successes"]) / len(x["history"]["successes"]), 
        reverse=False
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

    axs[0].set_title('Reward Distribution per Nr of agents', fontsize=14)
    axs[0].set_ylabel('Episode Return', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.4, axis='y')
 
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
    
    plt.savefig("ZSG.svg", format="svg")
    plt.show()


plot_comparison(results)

ray.shutdown()
