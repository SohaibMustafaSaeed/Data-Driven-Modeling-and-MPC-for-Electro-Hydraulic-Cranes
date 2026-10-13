import argparse

from common.project import ROOT


def main():
    parser = argparse.ArgumentParser(description='Open bundled full-recording figure exports without rerunning models or MPC')
    parser.add_argument('--experiment', choices=['all', 'models', 'reachable', 'recorded', 'iterations', 'neural', 'neural-latency', 'teacher', 'improvement'], default='all')
    parser.add_argument('--no-show',action='store_true',help='Validate selected figure files without opening windows')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    groups = {'models': ['model_comparison_2s_angles.png', 'model_comparison_2s_velocities.png', 'prediction_error_vs_lead.png'],
              'reachable': ['mpc_reachable_tip_path.png', 'mpc_reachable_tip_tracking.png', 'mpc_reachable_both_booms.png'],
              'recorded': ['mpc_recorded_tip_path.png', 'mpc_recorded_tip_tracking.png', 'recorded_replay_vs_mpc.png'],
              'iterations': ['slsqp_iteration_study.png'],
              'teacher':['teacher_cartesian_workspace_coverage.png','teacher_teacher_command_pair_distribution.png'],
              'neural':['neural_closed_loop_tip_trajectory_xy_or_xz.png','neural_episode_rmse_distribution.png','neural_representative_command_episode.png'],
              'neural-latency':['neural_latency_distribution.png','neural_accuracy_latency_comparison.png'],
              'improvement':['improvement_episode_rmse_distribution_baseline_vs_improved.png']}
    for group, files in groups.items():
        if args.experiment not in ('all', group):
            continue
        for filename in files:
            path=ROOT/'docs/figures'/filename
            if not path.is_file():raise FileNotFoundError(path)
            if args.no_show:
                print(path.relative_to(ROOT));continue
            fig, ax = plt.subplots(figsize=(13, 8), layout='constrained')
            ax.imshow(plt.imread(ROOT / 'docs/figures' / filename))
            ax.axis('off')
            fig.canvas.manager.set_window_title(filename)
    print('These are static publication exports. Regenerated local result arrays support raw-sample interactive time plots. Close windows to finish.', flush=True)
    if not args.no_show:plt.show(block=True)


if __name__ == '__main__':
    main()
