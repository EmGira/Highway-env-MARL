from optuna import load_study
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances


study = load_study(
    study_name='PPO_Study_2026-04-13_Run_0', 
    storage="sqlite:///optuna_highway_results.db"
)

fig_history = plot_optimization_history(study)
fig_parallel = plot_parallel_coordinate(study)
fig_importance = plot_param_importances(study)


fig_history.write_html("optuna_history.html")
fig_parallel.write_html("optuna_parallel.html")
fig_importance.write_html("optuna_importance.html")

print("Graphs succesfully saved")


