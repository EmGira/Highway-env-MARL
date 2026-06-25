import os
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PlotTraining import Helper

def plot_folder_seeds(folder_path, run_name, target_metrics, output_file, max_iters):
    
    files = []
    for root, dirs, files_in_dir in os.walk(folder_path):
        if 'params.json' in files_in_dir:
            files.append(root)
            
    if not files:
        print(f"No runs (with params.json) found in {folder_path}")
        return

    print(f"Found {len(files)} runs in {folder_path}:")
    for f in files:
        print(f"  - {f}")
    

    runs_dict = Helper.load_runs(files, target_metrics=target_metrics, max_iterations=max_iters)
    
    n_metrics = len(target_metrics)
    
   
    if n_metrics == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

    formal_names = {
        "return": "Average Episode Return",
        "vf_loss": "Value Function Loss",
        "policy_loss": "Policy Loss",
        "entropy": "Policy Entropy",
        "actor_loss": "Actor Loss",
        "alpha_loss": "Alpha Loss",
        "critic_loss": "Critic Loss"
    }

    for i, tm in enumerate(target_metrics):
        ax = axes[i]
        formal_ylabel = formal_names.get(tm, tm.capitalize())
        
        is_bottom_row = (n_metrics != 4 and i == n_metrics - 1) or (n_metrics == 4 and i >= 2)
        x_label = 'Environment Steps (Millions)' if is_bottom_row else ''
        
        # Aggregate data across all seeds
        all_y = []
        all_x = []
        for path in files:
      
            smoothed_y = Helper.smooth(runs_dict[path][tm], weight=0.85)
            x_vals = runs_dict[path]['num_samples']
            
            # Replace Nones with NaNs for numpy operations
            all_y.append([np.nan if v is None else v for v in smoothed_y])
            all_x.append([np.nan if v is None else v for v in x_vals])
            
        if all_y and len(all_y[0]) > 0:
            # Truncate to the minimum length in case runs stopped at different iterations
            min_len = min(len(arr) for arr in all_y)
            y_np = np.array([arr[:min_len] for arr in all_y])
            x_np = np.array([arr[:min_len] for arr in all_x])
            
            # Calculate mean and standard deviation
            mean_y = np.nanmean(y_np, axis=0)
            std_y = np.nanstd(y_np, axis=0)
            mean_x = np.nanmean(x_np, axis=0)
            
            # Plot the mean line
            line = ax.plot(mean_x, mean_y, linewidth=2.5, label=f"{run_name} (Mean \u00B1 Std)")[0]
            
            # Plot the standard deviation shadow
            ax.fill_between(mean_x, mean_y - std_y, mean_y + std_y, color=line.get_color(), alpha=0.2)

        ax.set_ylabel(formal_ylabel, fontweight='bold')
        ax.set_xlabel(x_label, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        ax.set_title(f'{formal_ylabel} Over Training', pad=10)

    fig.suptitle(f'{run_name} Performance Across {len(files)} Runs', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_file, format="svg")
    print(f"Saved plot to {output_file}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder containing the runs to aggregate")
    parser.add_argument("--name", type=str, required=True, help="Name of the algorithm/experiment (e.g. MAPPO)")
    parser.add_argument("--output", type=str, default=None, help="Output file name (defaults to <name>_seeds_plot.svg)")
    parser.add_argument("--metrics", nargs="+", default=["return", "vf_loss", "entropy"], help="Metrics to plot")
    parser.add_argument("--max_iters", type=int, default=0, help="Max iterations to plot. Use 0 for unlimited.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: Directory {args.folder} does not exist.")
        sys.exit(1)

    out_file = args.output if args.output else f"{args.name}_seeds_plot.svg"
    max_iters = None if args.max_iters <= 0 else args.max_iters

    plot_folder_seeds(args.folder, args.name, args.metrics, out_file, max_iters)
