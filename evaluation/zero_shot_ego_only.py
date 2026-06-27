import sys
import os
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_folder)


import ray
import numpy as np
import matplotlib.pyplot as plt
import torch

from configs.intersection.IntersectionConfigs import get_improved_Simple_config, get_ego_only_config

from utils.evaluation_utils import load_policy_or_module, create_eval_env, compute_actions, compute_actions_legacy

import numpy as np
import matplotlib.pyplot as plt


@ray.remote(num_cpus=1)
def distributed_evaluate_worker(checkpoint_path, env_config, num_episodes, start_index=0, seed_base=None):
    model_or_policy, stack_type = load_policy_or_module(checkpoint_path)
    env = create_eval_env(stack_type, env_config, ENV_ID, render_mode=None, inference_mode=False)

    worker_history = {
        "rewards": [],
        "crashes": [],
        "successes": []
    }

    for ep in range(num_episodes):
        # Common Random Numbers: the global episode index determines the seed,
        # so every policy/scenario faces the same set of initial conditions
        # (paired comparison). seed_base=None falls back to unseeded resets.
        if seed_base is None:
            obs, info = env.reset()
        else:
            obs, info = env.reset(seed=seed_base + start_index + ep)
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


def run_distributed_evaluation(policy_name, checkpoint_path, env_config, total_episodes=100, num_workers=4, seed_base=None):
    print(f"\n{'='*50}")
    print(f"Validating: {policy_name}")
    print(f"{'='*50}")


    episodes_per_worker = total_episodes // num_workers

    # Disjoint, contiguous episode-index ranges per worker so the global index
    # (start_index + ep) is unique and identical across scenarios for a given seed_base.
    futures = [
        distributed_evaluate_worker.remote(
            checkpoint_path, env_config, episodes_per_worker,
            start_index=w * episodes_per_worker, seed_base=seed_base
        )
        for w in range(num_workers)
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



def collect_data(output_dir):
    import json
    ray.init(ignore_reinit_error=True)
    results = []
    for scenario in scenarios:
        history, mean, std = run_distributed_evaluation(
            scenario["name"], scenario["checkpoint"], scenario["config"], NUM_TEST_EPISODES, NUM_WORKERS,
            seed_base=SEED_BASE
        )
        results.append({
            "name": scenario["name"],
            "history": history,
            "mean": float(mean),
            "std": float(std)
        })
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    ray.shutdown()
    print(f"Data saved to {os.path.join(output_dir, 'results.json')}")


def plot_comparison(output_dir):
    import json
    data_path = os.path.join(output_dir, "results.json")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Run with --collect first.")
        return
        
    with open(data_path, "r") as f:
        results = json.load(f)

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
    fig.suptitle('Zero-Shot Generalization - Intersection', fontsize=18, fontweight='bold')



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

    axs[1].set_title(f'Final Evaluation Metrics ({NUM_TEST_EPISODES} Episodes)', fontsize=14)
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
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ZSG.svg"), format="svg")
    plt.show()





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zero-Shot Generalization Evaluation")
    parser.add_argument("--collect", action="store_true", help="Run evaluation and collect data")
    parser.add_argument("--plot", action="store_true", help="Plot previously collected data")
    parser.add_argument("--output_dir", type=str, default="./data_zero_shot", help="Directory to save/load data")
    parser.add_argument("--seed_base", type=int, default=0,
                        help="Base seed for paired (Common Random Numbers) evaluation. "
                             "Episode i uses seed_base+i for every scenario, so all policies "
                             "face identical initial conditions. Use -1 for unseeded random episodes.")

    args = parser.parse_args()

    # -1 -> unseeded (legacy behaviour); otherwise paired seeding across scenarios
    SEED_BASE = None if args.seed_base < 0 else args.seed_base
    

    

    MAX_NR_AGENTS = 8
    NUM_TEST_EPISODES = 200
    NUM_WORKERS = 7 

    ENV_ID = "customIntersection-env-v0"

    def get_base_config():
        config = get_ego_only_config(3)
        config["simulation_frequency"] = 15
        config["randomize_controlled_vehicles"] = False
        return config

    #./A-checkpoints/TEST/3agents-MAPPO/MAPPO_2/ID_a2e6e_00000/checkpoint_000005
    #./A-checkpoints/TEST/3agents-IPPO/PPO_0/ID_4cd2d_00000/checkpoint_000005
    checkpoint = "./A-checkpoints/TEST/3agents-IPPO/PPO_0/ID_4cd2d_00000/checkpoint_000005"

    scenarios = [
        {
            "name": f"{nAgents} agents",
            "checkpoint": os.path.abspath(checkpoint),
            "config": {**get_base_config(), "controlled_vehicles" : nAgents } 
        }   for nAgents in range(3, MAX_NR_AGENTS+1)
    
    ]


    if not args.collect and not args.plot:
        print("Please specify --collect to gather data or --plot to create graphs.")
        parser.print_help()
    
    if args.collect:
        collect_data(args.output_dir)
        
    if args.plot:
        plot_comparison(args.output_dir)

