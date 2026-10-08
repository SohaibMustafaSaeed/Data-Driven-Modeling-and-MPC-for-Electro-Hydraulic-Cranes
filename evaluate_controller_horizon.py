import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from common.project import ROOT, RAW_DATA_DIR, read_report
from common.data import INPUT_COLS, OUTPUT_COLS, STATE_INDICES
from common.evaluation import Metrics
from common.horizons import controller_protocol
from common.recording_view import plot_recording_pair, scalar_source


def interval_metrics(cache, prediction, mean, std, block, intervals):
    horizon = block * intervals
    grouped = [Metrics(std) for _ in range(intervals)]
    endpoints = [Metrics(std) for _ in range(intervals)]
    for first in range(0, len(prediction), horizon):
        anchor = np.asarray(cache[first + 1, list(STATE_INDICES)], float) * std + mean
        for lead in range(intervals):
            begin = first + lead * block
            end = min(begin + block, len(prediction))
            if begin >= end:
                break
            actual = np.asarray(cache[begin + 2:end + 2][:, list(STATE_INDICES)], float) * std + mean
            forecast = np.asarray(prediction[begin:end])
            baseline = np.broadcast_to(anchor, actual.shape)
            grouped[lead].add(actual, forecast, baseline)
            if end - begin == block:
                endpoints[lead].add(actual[-1:], forecast[-1:], baseline[-1:])
    frames = []
    for scope, meters in [('all_samples_in_interval', grouped), ('exact_interval_endpoint', endpoints)]:
        for lead, meter in enumerate(meters, 1):
            if not meter.n:
                continue
            frame = meter.table()
            frame.insert(0, 'controller_lead', lead)
            frame.insert(0, 'scope', scope)
            frame['predictions'] = meter.n
            frame['R2_deficit'] = 1 - frame.R2
            frame['RMSE_relative_to_persistence'] = frame.RMSE / frame.persistence_RMSE.replace(0, np.nan)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Validate dynamics over a real controller-time horizon across the entire retained CSV')
    parser.add_argument('--controller-config', type=Path, default=ROOT / 'MPC offline Model/results/full_csv_patu_100ms/config.json')
    parser.add_argument('--controller-horizon', type=int, default=20)
    parser.add_argument('--test-file', type=Path, default=Path('Simulation_Output_53.csv'))
    parser.add_argument('--saved-only', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    models = [('GRU', 'gru', 'GRU Latent'), ('LSTM', 'LSTM', 'LSTM'), ('Transformer', 'transformer', 'Transformers'), ('Compact Recurrent', 'trainCompactLatent', 'Compact Recurrent')]
    runs, global_tables, leads, summaries, coverage = [], [], [], [], []
    protocol = None
    for name, module, folder in models:
        trainer = importlib.import_module(module)
        checkpoint = ROOT / folder / 'artifacts/best_model.pt'
        model, ck = trainer.load_bundle(checkpoint)
        current_protocol = controller_protocol(args.controller_config, float(ck['scaler']['sampling_interval_seconds']), args.controller_horizon)
        if protocol is None:
            protocol = current_protocol
            print(json.dumps(protocol, indent=2), flush=True)
        elif current_protocol != protocol:
            raise ValueError('Model/controller time grids differ')
        fp = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        directory = ROOT / folder / 'results' / args.test_file.stem / f'checkpoint_{fp[:16]}' / f'recursive_reinit{protocol["internal_horizon_steps"]}'
        if not (directory / 'evaluation.json').is_file():
            if args.saved_only:
                raise FileNotFoundError(f'Missing controller-time predictions for {name}; omit --saved-only to compute them')
            command = [sys.executable, str(ROOT / folder / 'test.py'), '--controller-config', str(args.controller_config),
                       '--controller-horizon', str(args.controller_horizon), '--test-file', str(args.test_file),
                       '--latency-repeats', '10', '--no-show', '--plot-points', '2000']
            print(f'Evaluating {name} over every retained sample, horizon={protocol["prediction_duration_seconds"]:g}s', flush=True)
            subprocess.run(command, check=True)
        report = read_report(directory / 'evaluation.json')
        requested_source = (args.test_file if args.test_file.is_absolute() else RAW_DATA_DIR / 'Test' / args.test_file).resolve()
        if Path(report['source_csv']).resolve() != requested_source:
            raise ValueError('Saved evaluations refer to a different recording')
        cache = np.load(report['cache'], mmap_mode='r')
        prediction = np.load(directory / 'predictions_physical.npy', mmap_mode='r')
        if report['checkpoint_sha256'] != fp or report['reinit_every'] != protocol['internal_horizon_steps'] or report['max_rows'] is not None:
            raise ValueError('Saved predictions do not match the requested checkpoint and full-recording horizon')
        if len(prediction) != len(cache) - 2 or not np.isfinite(report['dt_s']) or report['dt_s'] != float(ck['scaler']['sampling_interval_seconds']):
            raise ValueError('Saved prediction coverage or sample time differs')
        full = json.loads((directory / 'recording_coverage.json').read_text())
        assert full['full_retained_recording']
        coverage.append(dict(model=name, **full))
        scaler = ck['scaler']
        if runs and any(not np.array_equal(scaler[key], runs[0][3][key]) for key in ('mean', 'std')):
            raise ValueError('Normalization differs between models')
        mean, std = (np.asarray(scaler[key])[list(STATE_INDICES)] for key in ('mean', 'std'))
        table = pd.read_csv(directory / 'metrics.csv')
        table.insert(0, 'model', name)
        table['R2_deficit'] = 1 - table.R2
        table['RMSE_relative_to_persistence'] = table.RMSE / table.persistence_RMSE.replace(0, np.nan)
        global_tables.append(table)
        lead_path = directory / f'ctrl_c{args.controller_horizon}_b{protocol["samples_per_control_interval"]}_v1.csv'
        if lead_path.is_file() and lead_path.stat().st_mtime_ns >= (directory / 'predictions_physical.npy').stat().st_mtime_ns:
            lead = pd.read_csv(lead_path)
        else:
            lead = interval_metrics(cache, prediction, mean, std, protocol['samples_per_control_interval'], args.controller_horizon)
            lead.to_csv(lead_path, index=False)
        lead.insert(0, 'model', name)
        lead['interval_end_seconds'] = lead.controller_lead * protocol['control_interval_seconds']
        assert (lead[lead.scope == 'all_samples_in_interval'].groupby('feature').predictions.sum() == len(prediction)).all()
        leads.append(lead)
        summaries.append(dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                              predictions=len(prediction), reset_blocks=report['latent_resets'],
                              controller_intervals=args.controller_horizon, control_interval_seconds=protocol['control_interval_seconds'],
                              horizon_seconds=protocol['prediction_duration_seconds'], internal_horizon_steps=protocol['internal_horizon_steps'],
                              mean_NRMSE_all_states=table.NRMSE_train_std.mean(), checkpoint_sha256=fp))
        runs.append((name, prediction, cache, scaler, report))
        print('\n' + name + ': all retained targets\n' + table.to_string(index=False, float_format=lambda v: f'{v:.9g}'), flush=True)
    out = ROOT / 'results/controller_horizon' / args.test_file.stem / f'c{args.controller_horizon}_b{protocol["samples_per_control_interval"]}'
    out.mkdir(parents=True, exist_ok=True)
    tables, figures = ROOT / 'docs/results', ROOT / 'docs/figures'
    prefix = f'controller{args.controller_horizon}'
    for filename, table in [('model_summary', pd.DataFrame(summaries)), ('all_state_metrics', pd.concat(global_tables)),
                            ('metrics_by_interval', pd.concat(leads)), ('recording_coverage', pd.DataFrame(coverage))]:
        table.to_csv(out / f'{filename}.csv', index=False)
        table.to_csv(tables / f'{prefix}_{filename}.csv', index=False)
    protocol.update(predictions_per_model=len(runs[0][1]), reset_blocks=summaries[0]['reset_blocks'],
                    last_block_predictions=len(runs[0][1]) % protocol['internal_horizon_steps'] or protocol['internal_horizon_steps'],
                    endpoint_policy='Only complete controller intervals scored as exact endpoints; partial tail scored in all-sample metrics',
                    mpc_planning_horizon_unchanged=True, checkpoints=pd.DataFrame(summaries)[['model', 'checkpoint_sha256']].to_dict(orient='records'))
    for destination in [out / 'protocol.json', tables / f'{prefix}_protocol.json']:
        destination.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg' if args.no_show else 'TkAgg')
    import matplotlib.pyplot as plt
    features = ['thetalift[rad]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
    for rows, suffix in [([0, 1], 'angles'), ([2, 3], 'velocities')]:
        fig, axes = plt.subplots(2, 4, figsize=(19, 7), layout='constrained')
        for row, index in enumerate(rows):
            feature = features[index]
            column, output_column = INPUT_COLS.index(feature), OUTPUT_COLS.index(feature)
            for k, (name, prediction, cache, scaler, report) in enumerate(runs):
                ax = axes[row, k]
                plot_recording_pair(ax, scalar_source(cache, column, 2, scaler['mean'][column], scaler['std'][column]),
                                    scalar_source(prediction, output_column), len(prediction), report['dt_s'],
                                    report['excluded_seconds'] + 2 * report['dt_s'], name, 5000)
                metric = global_tables[k][global_tables[k].feature == feature].iloc[0]
                ax.set_title(f'{name}: {feature}\nRMSE {metric.RMSE:.5g} | model/hold RMSE {metric.RMSE_relative_to_persistence:.3f}', fontsize=10)
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
        fig.suptitle(f'Whole retained CSV: {args.controller_horizon} × {protocol["control_interval_seconds"] * 1000:g} ms = {protocol["prediction_duration_seconds"]:g}s forecasts; measured reset every forecast block')
        fig.savefig(figures / f'{prefix}_full_csv_{suffix}.png', dpi=150)
    for scope, suffix in [('all_samples_in_interval', 'interval_errors'), ('exact_interval_endpoint', 'endpoint_errors')]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
        for feature, ax in zip(features, axes.flat):
            for name, lead in zip([name for name, _, _ in models], leads):
                view = lead[(lead.feature == feature) & (lead.scope == scope)]
                ax.plot(view.interval_end_seconds, view.RMSE, label=name)
            ax.plot(view.interval_end_seconds, view.persistence_RMSE, '--', color='black', label='Hold initial measured state')
            ax.set(title=feature, xlabel='Prediction lead [s]', ylabel='Physical-unit RMSE')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f'{scope}: errors versus persistence; every forecast block across the entire CSV')
        fig.savefig(figures / f'{prefix}_{suffix}.png', dpi=150)
    print('\nController-time summary:\n' + pd.DataFrame(summaries).to_string(index=False), flush=True)
    print('Saved controller-time validation: ' + str(out), flush=True)
    if not args.no_show:
        plt.show(block=True)
    plt.close('all')


if __name__ == '__main__':
    main()
