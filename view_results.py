import argparse

from common.project import ROOT


def main():
    parser = argparse.ArgumentParser(description='Open bundled full-recording figure exports without rerunning models or MPC')
    parser.add_argument('--experiment', choices=['all', 'models', 'reachable', 'recorded', 'iterations'], default='all')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    groups = {'models': ['model_comparison_2s_angles.png', 'model_comparison_2s_velocities.png', 'prediction_error_vs_lead.png'],
              'reachable': ['mpc_reachable_tip_path.png', 'mpc_reachable_tip_tracking.png', 'mpc_reachable_both_booms.png'],
              'recorded': ['mpc_recorded_tip_path.png', 'mpc_recorded_tip_tracking.png', 'recorded_replay_vs_mpc.png'],
              'iterations': ['slsqp_iteration_study.png']}
    for group, files in groups.items():
        if args.experiment not in ('all', group):
            continue
        for filename in files:
            fig, ax = plt.subplots(figsize=(13, 8), layout='constrained')
            ax.imshow(plt.imread(ROOT / 'docs/figures' / filename))
            ax.axis('off')
            fig.canvas.manager.set_window_title(filename)
    print('These are static publication exports. Regenerated local result arrays support raw-sample interactive time plots. Close windows to finish.', flush=True)
    plt.show(block=True)


if __name__ == '__main__':
    main()
