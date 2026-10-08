import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.project import ROOT, read_report
from common.data import OUTPUT_COLS, STATE_INDICES, INPUT_COLS
from common.recording_view import plot_recording_pair, scalar_source


def lead_metrics(cache, prediction, mean, std, horizon):
    counts = np.zeros(horizon, np.int64)
    average = np.zeros((horizon, 22))
    m2, sse, sae, maximum = (np.zeros_like(average) for _ in range(4))
    size = horizon * max(1, 20000 // horizon)
    for begin in range(0, len(prediction), size):
        end = min(begin + size, len(prediction))
        actual = np.asarray(cache[begin + 2:end + 2][:, list(STATE_INDICES)], float) * std + mean
        error = np.asarray(prediction[begin:end]) - actual
        full = len(actual) // horizon * horizon
        if full:
            values = actual[:full].reshape(-1, horizon, 22)
            errors = error[:full].reshape(-1, horizon, 22)
            count = len(values)
            local_mean = values.mean(0)
            delta = local_mean - average
            m2 += ((values - local_mean) ** 2).sum(0) + delta ** 2 * (counts * count / (counts + count))[:, None]
            average += delta * (count / (counts + count))[:, None]
            counts += count
            sse += (errors ** 2).sum(0)
            sae += abs(errors).sum(0)
            maximum = np.maximum(maximum, abs(errors).max(0))
        for j in range(len(actual) - full):
            delta = actual[full + j] - average[j]
            counts[j] += 1
            average[j] += delta / counts[j]
            m2[j] += delta * (actual[full + j] - average[j])
            sse[j] += error[full + j] ** 2
            sae[j] += abs(error[full + j])
            maximum[j] = np.maximum(maximum[j], abs(error[full + j]))
    frames = []
    for j in range(horizon):
        if not counts[j]:
            continue
        ratio = np.full(22, np.nan)
        np.divide(sse[j], m2[j], out=ratio, where=m2[j] / counts[j] > np.finfo(float).eps * np.maximum(1., average[j] ** 2))
        rmse = np.sqrt(sse[j] / counts[j])
        frames.append(pd.DataFrame(dict(feature=OUTPUT_COLS, prediction_step=j + 1, predictions=counts[j],
                                       RMSE=rmse, MAE=sae[j] / counts[j], R2=1 - ratio,
                                       max_abs_error=maximum[j], NRMSE_train_std=rmse / std)))
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reinit-every', type=int, default=200)
    parser.add_argument('--test-file', type=Path, default=Path('Simulation_Output_53.csv'))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.reinit_every < 1:
        parser.error('A positive reset horizon is required')
    import matplotlib
    matplotlib.use('Agg' if args.no_show else 'TkAgg')
    import matplotlib.pyplot as plt

    horizon = args.reinit_every
    out = ROOT / 'results' / f'reset{horizon}' / args.test_file.stem
    out.mkdir(parents=True, exist_ok=True)
    figures, tables = ROOT / 'docs/figures', ROOT / 'docs/results'
    models = [('GRU', 'GRU Latent'), ('LSTM', 'LSTM'), ('Transformer', 'Transformers'), ('Compact Recurrent', 'Compact Recurrent')]
    features = ['thetalift[rad]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
    runs, metrics, summaries, coverage, leads = [], [], [], [], []
    for name, folder in models:
        checkpoint = ROOT / folder / 'artifacts/best_model.pt'
        fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        directory = ROOT / folder / 'results' / args.test_file.stem / f'checkpoint_{fingerprint[:16]}' / f'recursive_reinit{horizon}'
        report = read_report(directory / 'evaluation.json')
        prediction = np.load(directory / 'predictions_physical.npy', mmap_mode='r')
        cache = np.load(report['cache'], mmap_mode='r')
        assert report['reinit_every'] == horizon and report['max_rows'] is None and len(prediction) == len(cache) - 2
        trainer = importlib.import_module({'GRU': 'gru', 'LSTM': 'LSTM', 'Transformer': 'transformer', 'Compact Recurrent': 'trainCompactLatent'}[name])
        model, ck = trainer.load_bundle(checkpoint)
        scaler = ck['scaler']
        mean, std = np.asarray(scaler['mean'])[list(STATE_INDICES)], np.asarray(scaler['std'])[list(STATE_INDICES)]
        table = pd.read_csv(directory / 'metrics.csv')
        table.insert(0, 'model', name)
        table['R2_deficit'] = 1 - table.R2
        table['RMSE_relative_to_persistence'] = table.RMSE / table.persistence_RMSE.replace(0, np.nan)
        metrics.append(table)
        lead = lead_metrics(cache, prediction, mean, std, horizon)
        lead.insert(0, 'model', name)
        lead.to_csv(directory / 'metrics_by_prediction_step.csv', index=False)
        leads.append(lead)
        full = json.loads((directory / 'recording_coverage.json').read_text())
        assert full['full_retained_recording']
        coverage.append(dict(model=name, **full))
        summaries.append(dict(model=name, parameters=sum(v.numel() for v in model.parameters()), predictions=len(prediction),
                              reset_steps=horizon, reset_blocks=report['latent_resets'], horizon_duration_ms=horizon * report['dt_s'] * 1000,
                              mean_NRMSE_all_states=table.NRMSE_train_std.mean(),
                              mean_NRMSE_mechanical_states=table[table.feature.isin(features)].NRMSE_train_std.mean(),
                              reset_horizon_median_us=report['horizon_latency']['median_us'],
                              reset_horizon_p95_us=report['horizon_latency']['p95_us'], checkpoint_sha256=fingerprint))
        runs.append((name, prediction, cache, scaler, report))
        print(f'Analyzed every sample and all {horizon} prediction leads: {name}', flush=True)
    summary = pd.DataFrame(summaries)
    for filename, table in [('model_summary', summary), ('all_state_metrics', pd.concat(metrics)),
                            ('metrics_by_prediction_step', pd.concat(leads)), ('recording_coverage', pd.DataFrame(coverage))]:
        table.to_csv(out / f'{filename}.csv', index=False)
        table.to_csv(tables / f'reset{horizon}_{filename}.csv', index=False)
    for rows, suffix in [([0, 1], 'angles'), ([2, 3], 'velocities')]:
        fig, axes = plt.subplots(2, 4, figsize=(19, 7), layout='constrained')
        for row, feature_index in enumerate(rows):
            feature = features[feature_index]
            column, output_column = INPUT_COLS.index(feature), OUTPUT_COLS.index(feature)
            for k, (name, prediction, cache, scaler, report) in enumerate(runs):
                ax = axes[row, k]
                plot_recording_pair(ax, scalar_source(cache, column, 2, scaler['mean'][column], scaler['std'][column]),
                                    scalar_source(prediction, output_column), len(prediction), report['dt_s'],
                                    report['excluded_seconds'] + 2 * report['dt_s'], name, 5000)
                metric = metrics[k][metrics[k].feature == feature].iloc[0]
                ax.set_title(f'{name}: {feature}\nRMSE {metric.RMSE:.5g} | 1 - R² {1 - metric.R2:.3e}', fontsize=10)
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
        fig.suptitle(f'Entire retained CSV: predict {horizon} steps ({summaries[0]["horizon_duration_ms"]:g} ms), then reset; {summaries[0]["reset_blocks"]:,} blocks')
        fig.savefig(figures / f'reset{horizon}_full_csv_{suffix}.png', dpi=150)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for feature, ax in zip(features, axes.flat):
        values = np.array([table[table.feature == feature].RMSE.iloc[0] for table in metrics])
        persistence = metrics[0][metrics[0].feature == feature].persistence_RMSE.iloc[0]
        ax.bar([name for name, _ in models], values, label='Model RMSE')
        ax.axhline(persistence, color='black', linestyle='--', label='Hold measured state: RMSE')
        ax.set_title(feature)
        ax.set_ylabel('Physical-unit RMSE')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'All retained samples: {horizon}-step forecasts versus persistence')
    fig.savefig(figures / f'reset{horizon}_mechanical_persistence.png', dpi=150)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for feature, ax in zip(features, axes.flat):
        for name, lead in zip([name for name, _ in models], leads):
            selected = lead[lead.feature == feature]
            ax.plot(selected.prediction_step, selected.RMSE, label=name)
        ax.set(title=feature, xlabel='Prediction lead [internal samples]', ylabel='RMSE')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Prediction error by lead; statistics use every block across the whole CSV')
    fig.savefig(figures / f'reset{horizon}_error_by_lead.png', dpi=150)
    provenance = dict(method=f'{horizon} recursive internal steps followed by measured-state and memory reset',
                      dataset=args.test_file.name, predictions=summaries[0]['predictions'],
                      horizon_duration_ms=summaries[0]['horizon_duration_ms'],
                      reset_blocks=summaries[0]['reset_blocks'], weights_unchanged=True,
                      R2_display='1 - R2 in scientific notation to expose near-one values',
                      trajectory_results='MPC trajectories are evaluated separately using the independent mechanical plant',
                      checkpoints=summary[['model', 'checkpoint_sha256']].to_dict(orient='records'))
    (tables / f'reset{horizon}_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    if not args.no_show:
        plt.show(block=True)
    plt.close('all')


if __name__ == '__main__':
    main()
