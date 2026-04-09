import ray
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from configs.intersection.IntersectionConfigs import get_busy_intersection_config, get_simple_multi_agent_config

ray.init(ignore_reinit_error=True)


def compute_actions(multi_rl_module, obs):

    with torch.no_grad():
        agents_actions = {}
        for agent_id, agent_obs in obs.items():
            ao = torch.from_numpy(agent_obs).float().unsqueeze(0)
            output = multi_rl_module[agent_id].forward_inference({"obs": ao})
            agents_actions[agent_id] = torch.argmax(output["action_dist_inputs"], dim=1).item()
    
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

ENV_CONFIG_A = get_busy_intersection_config(num_agents=NR_AGENTS)
ENV_CONFIG_A["spawn_points"] = ["0", "1"] 
ENV_CONFIG_A["multi_destinations"] = ["o1", "o0"] 
ENV_CONFIG_A["simulation_frequency"] = 15

ENV_CONFIG_B = get_busy_intersection_config(num_agents=NR_AGENTS)
ENV_CONFIG_B["spawn_points"] = ["3", "1"] 
ENV_CONFIG_B["multi_destinations"] = ["o0", "o3"] 
ENV_CONFIG_B["simulation_frequency"] = 15

CHECKPOINT_PATH_A = os.path.abspath(
    # "./A-checkpoints/SimpleConfig_A_Best_500iter/Run/PPO_Batch_2048-lr_1e-05_ID_e7303_00000/checkpoint_000037"
     "./A-checkpoints/BusyConfig_A_Best_500iter/RunID_0/PPO_Batch_2048-lr_1e-05_ID_37702_00000/checkpoint_000036")  
CHECKPOINT_PATH_B = os.path.abspath(
    # "./A-checkpoints/SimpleConfig_A_Best_500iter/Run/PPO_Batch_2048-lr_1e-05_ID_e7303_00000/checkpoint_000037"
     "./A-checkpoints/BusyConfig_A_Best_500iter/RunID_0/PPO_Batch_2048-lr_1e-05_ID_37702_00000/checkpoint_000036")  



NUM_TEST_EPISODES = 200
NUM_WORKERS = 5 

Name_A = "POLICY A on ENV_CONFIG A"
history_A, mean_A, std_A = run_distributed_evaluation(
    Name_A, CHECKPOINT_PATH_A, ENV_CONFIG_A, NUM_TEST_EPISODES, NUM_WORKERS
)

Name_B = "POLICY A on ENV_CONFIG B"
history_B, mean_B, std_B = run_distributed_evaluation(
    Name_B, CHECKPOINT_PATH_B, ENV_CONFIG_B, NUM_TEST_EPISODES, NUM_WORKERS
)






def plot_comparison(hist_A, hist_B, policies, mean_A, std_A, mean_B, std_B):
    episodes = np.arange(1, len(hist_A["crashes"]) + 1)
    
    crash_rate_A = np.cumsum(hist_A["crashes"]) / episodes * 100
    crash_rate_B = np.cumsum(hist_B["crashes"]) / episodes * 100
    
    success_rate_A = np.cumsum(hist_A["successes"]) / episodes * 100
    success_rate_B = np.cumsum(hist_B["successes"]) / episodes * 100

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    fig.suptitle('Zero-Shot Generalization: Policy A on A vs Policy A on B', fontsize=16, fontweight='bold')

    axs[0].plot(episodes, hist_A["rewards"], label=policies[0], marker='o', color='green', alpha=0.7)
    axs[0].plot(episodes, hist_B["rewards"], label=policies[1], marker='s', color='red',  alpha=0.7)
    axs[0].set_title('Reward per episode')
    axs[0].set_ylabel('ep reward')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    
    axs[0].axhline(y=mean_A, color='green', linestyle='-', label=f'Mean A ({mean_A:.1f})')
    axs[0].fill_between(episodes, mean_A - std_A, mean_A + std_A, color='green', alpha=0.2)

    axs[0].axhline(y=mean_B, color='red', linestyle='-', label=f'Mean B ({mean_B:.1f})')
    axs[0].fill_between(episodes, mean_B - std_B, mean_B + std_B, color='red', alpha=0.2)

    axs[0].legend()

    axs[1].plot(episodes, crash_rate_A, label=policies[0], color='green', linewidth=2)
    axs[1].plot(episodes, crash_rate_B, label=policies[1], color='red', linewidth=2)
    axs[1].set_title('Cumulative crash rate (%)')
    axs[1].set_ylabel('Crash Rate (%)')
    axs[1].set_ylim([-5, 105])
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    axs[2].plot(episodes, success_rate_A, label=policies[0], color='green', linewidth=2)
    axs[2].plot(episodes, success_rate_B, label=policies[1], color='red', linewidth=2)
    axs[2].set_title('Cumulative success rate (%)')
    axs[2].set_xlabel('episode')
    axs[2].set_ylabel('Success Rate (%)')
    axs[2].set_ylim([-5, 105])
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("comparison_results.svg", format="svg")
    plt.show()


Name_A = "Policy A"
Name_B = "Policy B"

plot_comparison(history_A, history_B, [Name_A, Name_B], mean_A, std_A, mean_B, std_B)

ray.shutdown()
