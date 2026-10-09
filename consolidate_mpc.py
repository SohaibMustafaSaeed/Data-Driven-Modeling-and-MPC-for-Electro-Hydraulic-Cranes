import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.project import ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(folder):
    cfg, summary = read(folder / 'config.json'), read(folder / 'summary.json')
    if not summary.get('mpc_executed'):
        raise ValueError(f'No completed MPC experiment: {folder}')
    full = cfg.get('full_recording', False)
    if full and not read(folder / 'recording_coverage.json').get('full_retained_recording'):
        raise ValueError(f'Incomplete full-recording result: {folder}')
    joint = pd.read_csv(folder / 'tracking_metrics.csv').set_index('feature')
    tip = pd.read_csv(folder / 'end_effector_metrics.csv').set_index('feature')
    history = pd.read_csv(folder / 'solver_history.csv')
    distance = summary['all_microstep_end_effector_distance'] if full else summary['kinematic_tracking']['end_effector']
    row = dict(reference_mode=cfg['reference_mode'], maxiter=cfg['arguments']['maxiter'], updates=len(history),
               metric_scope='all internal samples' if full else 'control endpoints', duration_s=cfg['actual_duration_seconds'])
    row.update({key[:-2] + '_mm': value * 1000 for key, value in distance.items() if key.endswith('_m')})
    for feature in joint.index:
        row[feature + ' RMSE'] = float(joint.loc[feature, 'RMSE'])
    for feature in tip.index:
        row[feature + ' RMSE'] = float(tip.loc[feature, 'RMSE'])
    for key in ['mean_solve_ms', 'median_solve_ms', 'p95_solve_ms', 'max_solve_ms', 'solver_success_percent',
                'lift_saturation_percent', 'jib_saturation_percent', 'improved_over_hold_percent']:
        row[key] = summary[key]
    row['deadline_reached_percent'] = float(np.mean(history.solve_seconds >= cfg['control_interval_seconds']) * 100)
    row['iteration_cap_percent'] = float(np.mean(history.iterations >= row['maxiter']) * 100)
    row['average_iterations'] = float(history.iterations.mean())
    row['mean_objective_calls'] = float(history.candidate_calls.mean())
    mismatch = pd.read_csv(folder / ('model_mismatch_control_endpoints.csv' if full else 'model_mismatch_metrics.csv'))
    for value in mismatch.itertuples():
        row['one_interval_mismatch ' + value.feature] = value.RMSE
    print('\n' + str(folder) + '\n' + pd.Series(row).to_string(), flush=True)
    if cfg['reference_mode'] == 'recorded':
        import sys
        sys.path.insert(0, str(ROOT / 'MPC offline Model'))
        from crane_kinematics import PlanarGeometry, distance_metrics, axis_metrics
        geometry = PlanarGeometry(**cfg['geometry'])
        with np.load(folder / 'mpc_mechanical_plant_arrays.npz') as arrays:
            reference = geometry.end_effector(arrays['recorded_reference_physical'][1:])
            replay = geometry.end_effector(arrays['recorded_held_replay_physical'][1:])
            actual = geometry.end_effector(arrays['plant_mechanical_physical'][1:])
        base, controlled = distance_metrics(reference, replay), distance_metrics(reference, actual)
        records = [dict(metric=key, replay_mm=base[key] * 1000, MPC_mm=controlled[key] * 1000,
                        improvement_percent=(base[key] - controlled[key]) / base[key] * 100 if base[key] else None) for key in base]
        print('\nRECORDED REPLAY VS MPC: matching control endpoints; positive = improvement\n' + pd.DataFrame(records).to_string(index=False), flush=True)
    return row, cfg


def main():
    parser = argparse.ArgumentParser(description='Summarize generated local MPC results; never runs optimization or training')
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--comparison', choices=['predictors', 'iterations', 'references'], default='references')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    items = [summarize(folder) for folder in args.runs]
    allowed = {'output_dir', 'checkpoint'} if args.comparison == 'predictors' else {'output_dir', 'maxiter'} if args.comparison == 'iterations' else {'output_dir', 'reference_mode', 'allow_reference_mismatch'}
    for _, cfg in items[1:]:
        for key, value in items[0][1]['arguments'].items():
            if key not in allowed and cfg['arguments'].get(key) != value:
                raise ValueError('Uncontrolled settings difference: ' + key)
        for key in ['geometry', 'command_bounds', 'mechanical_checkpoint_sha256', 'source_csv', 'selected_retained_row']:
            if cfg[key] != items[0][1][key]:
                raise ValueError('Uncontrolled condition difference: ' + key)
    table = pd.DataFrame([row for row, _ in items], index=[folder.name for folder in args.runs])
    print('\nCOMPARISON\n' + table.select_dtypes(include='number').T.to_string(), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
    print('Different reference modes are different targets. No further experiment is launched.', flush=True)


if __name__ == '__main__':
    main()
