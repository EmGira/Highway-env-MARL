import matplotlib.pyplot as plt
import os
import json
from pprint import pprint
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass

#font sizes
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11
})

class Helper:

    @staticmethod
    def smooth(scalars, weight=0.85):
       
        if not scalars:
            return []
        
      
        first_val = 0
        for s in scalars:
            if s is not None:
                first_val = s
                break
                
        last = first_val
        smoothed = []
        for point in scalars:
            if point is None:
                smoothed.append(None)
                continue
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed

    @staticmethod
    def load_runs(paths, target_metrics=["return"]):
        runs_dict = {}

        for path in paths:
            runs_dict[path] = {'num_samples': []}

            for tm in target_metrics:
                runs_dict[path][tm] = []

            with open(f'{path}/params.json') as params_json:
                json_data = json.load(params_json)
                uses_new_api_stack = json_data.get('enable_rl_module_and_learner', False)



            steps_key = 'ray/tune/num_env_steps_sampled_lifetime' if uses_new_api_stack else 'ray/tune/num_env_steps_sampled'
           
            event_files = sorted([os.path.join(path, f) for f in os.listdir(path) if 'events.out.tfevents' in f])
            data_by_step = {}

            for ef in event_files:
                ea = EventAccumulator(ef)
                ea.Reload()
                tags = ea.Tags()['scalars']


                curr_steps_key = steps_key
                if curr_steps_key not in tags:
                    if 'ray/tune/info/num_env_steps_sampled' in tags:
                        curr_steps_key = 'ray/tune/info/num_env_steps_sampled'
                if curr_steps_key not in tags:
                    continue

                steps = ea.Scalars(curr_steps_key)
                step_dict = {s.step: s.value for s in steps}



                for tm in target_metrics:
                    
                    #find correct metric key depending on if the run utilizes old or new api stack
                    curr_metric_key = None
                    if tm == "return":
                        candidates = ['ray/tune/env_runners/episode_return_mean', 'ray/tune/episode_reward_mean', 'ray/tune/env_runners/episode_reward_mean']
                    elif tm == "vf_loss":
                        candidates = ['ray/tune/learners/shared_policy/vf_loss', 'ray/tune/info/learner/shared_policy/learner_stats/vf_loss', 'ray/tune/info/learner/default_policy/learner_stats/vf_loss']
                    elif tm == "policy_loss":
                        candidates = ['ray/tune/learners/shared_policy/policy_loss', 'ray/tune/info/learner/shared_policy/learner_stats/policy_loss']
                    elif tm == "entropy":
                        candidates = ['ray/tune/learners/shared_policy/entropy', 'ray/tune/info/learner/shared_policy/learner_stats/entropy']
                    else:
                        candidates = [tm]

                    for cand in candidates:
                        if cand in tags:
                            curr_metric_key = cand
                            break



                    if curr_metric_key and curr_metric_key in tags:
                        metric_data = ea.Scalars(curr_metric_key)
                        metric_dict = {m.step: m.value for m in metric_data}

                        #this correctly stiches data from two .tfevents, when a run has been stopped and resumed
                        for iteration, val in metric_dict.items():
                            if iteration in step_dict:
                                if iteration not in data_by_step:
                                    data_by_step[iteration] = {'s': step_dict[iteration]}
                                data_by_step[iteration][tm] = val


            sorted_iterations = sorted(data_by_step.keys())
            num_samples = []
            metrics_arrays = {tm: [] for tm in target_metrics}

            for it in sorted_iterations:
                row = data_by_step[it]
                num_samples.append(float(row['s']) / 1_000_000.0)

                for tm in target_metrics:
                    val = row.get(tm, None)

                    if val is None:
                        metrics_arrays[tm].append(None)
                    else:
                        metrics_arrays[tm].append(round(float(val), 4))


            runs_dict[path]['num_samples'] = num_samples[:100]

            for tm in target_metrics:
                runs_dict[path][tm] = metrics_arrays[tm][:100]


        return runs_dict
    

    @staticmethod
    def my_plotter(ax, data1, data2, param_dict, ylabel, xlabel, smooth_weight=0.85):
        
        color = ax.plot(data1, data2, alpha=0.3, linewidth=1.5)[0].get_color()
        
        
        smoothed_data2 = Helper.smooth(data2, weight=smooth_weight)
        
        
        ax.plot(data1, smoothed_data2, color=color, linewidth=2.5, **param_dict)

        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlabel(xlabel, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        return ax


if __name__ == "__main__":
        
    files = ['./A-checkpoints/TEST/6agents-MAPPO/MAPPO_0/ID_8fee6_00000', './A-checkpoints/TEST/6agents-IPPO/IPPO_0/ID_56159_00000']
    labels = ['MAPPO', 'IPPO']


    TARGET_METRICS = ["return", "vf_loss", "entropy"]
    
    runs_dict = Helper.load_runs(files, target_metrics=TARGET_METRICS)
    
    n_metrics = len(TARGET_METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]

   
    formal_names = {
        "return": "Average Episode Return",
        "vf_loss": "Value Function Loss",
        "policy_loss": "Policy Loss",
        "entropy": "Policy Entropy"
    }

    for i, tm in enumerate(TARGET_METRICS):
        ax = axes[i]
        
        formal_ylabel = formal_names.get(tm, tm.capitalize())
        
        
        x_label = 'Environment Steps (Millions)' if i == len(TARGET_METRICS) - 1 else ''
        
        for path, label in zip(files, labels):  
            Helper.my_plotter(ax, 
                              runs_dict[path]['num_samples'], 
                              runs_dict[path][tm], 
                              {'label': label}, 
                              ylabel=formal_ylabel, 
                              xlabel=x_label)
        
        ax.set_title(f'{formal_ylabel} Over Training', pad=10)

  
    fig.suptitle('Multi-Agent Highway Intersection (3 Agents)', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("temp.svg", format="svg")
