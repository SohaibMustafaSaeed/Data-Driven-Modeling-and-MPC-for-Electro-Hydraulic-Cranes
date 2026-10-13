import argparse
import json
from pathlib import Path

import pandas as pd

from common.project import ROOT


def main():
    parser = argparse.ArgumentParser(description='Print bundled lightweight publication results without the dataset or optimization')
    parser.add_argument('--section', choices=['all', 'models', 'dynamics', 'predictors', 'iterations', 'mpc', 'teacher', 'neural', 'improvement', 'final', 'replay'], default='all')
    args = parser.parse_args()
    tables = ROOT / 'docs/results'
    files = {'models': 'model_mechanical_comparison_2s.csv', 'predictors': 'mpc_predictor_comparison.csv',
             'iterations': 'slsqp_iteration_study.csv', 'mpc': 'final_gru_reference_comparison.csv', 'replay': 'recorded_replay_vs_mpc.csv',
             'teacher': 'teacher_dataset_summary.csv', 'neural': 'neural_controller_results.csv', 'improvement': 'improvement_study_results.csv'}
    for section, filename in files.items():
        selected='models' if args.section=='dynamics' else args.section
        if selected not in ('all', section, 'final'):
            continue
        print('\n' + section.upper(), flush=True)
        table = pd.read_csv(tables / filename)
        if section in ('predictors', 'mpc'):
            table = table.set_index('predictor' if section == 'predictors' else 'reference_mode').select_dtypes(include='number').T
            if args.section=='final':
                table=table.loc[[key for key in ['predictor_parameters','tip_position_RMSE_mm','tip_position_MAE_mm','tip_position_max_error_mm','tip_position_final_error_mm','mean_solve_ms','native_strict_deadline_miss_percent'] if key in table.index]]
        if section in ('neural','improvement'):
            if section=='neural':
                table=table[table.controller.isin(['MPC','MLP','GRU policy'])].copy()
                table.loc[table.controller=='MPC','parameters']=4098
                raw=json.loads((ROOT/'Neural Controller/results/imitation/imitation_metrics.json').read_text())
                table['command_RMSE_V']=table.controller.map({key:value['original_actual_targets']['combined_RMSE_V'] for key,value in raw.items()})
            columns=[c for c in ['controller','parameters','command_RMSE_V','tip_position_RMSE_mm','tip_position_MAE_mm','median_episode_RMSE_mm','p95_episode_RMSE_mm','worst_episode_RMSE_mm','mean_latency_ms','mean_ms','p99_latency_ms','p99_ms','deadline_miss_percent'] if c in table]
            table=table[columns]
        print(table.to_string(index=section in ('predictors','mpc'),float_format=lambda value: f'{value:.8g}',na_rep='-'), flush=True)
    if args.section in ('all','final','neural','improvement'):
        neural=ROOT/'Neural Controller'
        chosen=json.loads((neural/'models/selected_controller.json').read_text())
        decision=json.loads((neural/'results/improvement_study/comparison/decision.json').read_text())
        print('\nFINAL SELECTED NEURAL CONTROLLER: baseline GRU policy, 6,898 parameters.',flush=True)
        print('Checkpoint:',neural/'models/gru/policy.pt',flush=True)
        print('GRU/MPC tracking RMSE ratio:',chosen['tracking_RMSE_ratio_to_MPC'],flush=True)
        print('Focused ablation:',decision['decision'],'; tracking change vs baseline:',-decision['improvement_percent'],'% degradation',flush=True)
        print('Scope: learned-model simulation and CPU computational feasibility; no hardware, Simulink or formal-stability validation.',flush=True)
    print('\nPrimary full-record MPC metrics score every internal sample. Replay comparisons use matching control endpoints.', flush=True)
    print('Two-second surrogate validation and the 200-ms MPC planning horizon are different experiments.', flush=True)


if __name__ == '__main__':
    main()
