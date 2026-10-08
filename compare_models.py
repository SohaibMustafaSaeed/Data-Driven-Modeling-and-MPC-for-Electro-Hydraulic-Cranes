import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from common.recording_view import recording_coverage, scalar_source, plot_recording_pair, time_metrics, plot_temporal_errors
import torch
from common.data import OUTPUT_COLS, STATE_INDICES
SCRIPT_DIR = Path(__file__).resolve().parent
from common.results import read_result
from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, read_report, manifest_signature

def latest_pair():
    return (ROOT / 'GRU Latent/artifacts/best_model.pt', ROOT / 'LSTM/artifacts/best_model.pt')

def result_root(checkpoint, stem, explicit=None, legacy_compact=False):
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    version = f'checkpoint_{fingerprint[:16]}'
    root = explicit or checkpoint.parent.parent / 'results' / stem
    reports = list(root.glob('*/evaluation.json'))
    if reports and all((json.loads(report.read_text()).get('checkpoint_sha256') == fingerprint or (legacy_compact and legacy_report_matches(report, checkpoint)) for report in reports)):
        return (root, fingerprint)
    return (root / version, fingerprint)

def legacy_report_matches(report_path, checkpoint):
    report = json.loads(report_path.read_text())
    return report.get('checkpoint_sha256') is None and Path(report.get('checkpoint', '')).resolve() == checkpoint.resolve() and (report_path.stat().st_mtime_ns >= checkpoint.stat().st_mtime_ns)

def ensure_results(root, checkpoint, architecture, scenarios, args):
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for tag, mode in scenarios:
        folder = root / tag
        files = ('evaluation.json', 'metrics.csv', 'predictions_physical.npy')
        if all(((folder / name).is_file() for name in files)):
            report = json.loads((folder / 'evaluation.json').read_text())
            legacy = architecture == 'compact' and legacy_report_matches(folder / 'evaluation.json', checkpoint)
            if not legacy and report.get('architecture') != architecture:
                raise ValueError(f'Cached architecture differs: {folder}')
            if not legacy and report.get('checkpoint_sha256') != fingerprint:
                raise ValueError(f'Cached results use a different checkpoint: {folder}')
            if report.get('max_rows') is not None and (not args.allow_prefix):
                raise ValueError(f'Prefix evaluation requires --allow-prefix: {folder}')
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'Incomplete result folder: {folder}; supply a fresh result folder for this model')
        if args.saved_only:
            raise FileNotFoundError(f'Missing saved evaluation: {folder}')
        command = [sys.executable, str(ROOT / 'common/evaluate.py'), '--architecture', architecture, '--checkpoint', str(checkpoint), '--output-dir', str(root), '--test-file', args.test_file, '--mode', 'measured' if mode == 'measured_one_step' else 'recursive', '--no-show']
        if tag.startswith('recursive_reinit'):
            command += ['--reinit-every', str(args.reinit_every)]
        print(f'Evaluating missing {architecture} / {tag}', flush=True)
        subprocess.run(command, check=True)

def architecture_budget(trainer, model):
    if getattr(trainer, 'ARCHITECTURE', None) == 'transformer':
        return trainer.compute_budget(model.config)
    cfg = model.config
    if trainer.FORMAT.startswith('compact_latent'):
        L, H = (cfg.latent_dim, cfg.hidden)
        macs = (48 + L) * L + (48 + L) * H + H * 22 + 48 * 22
        return dict(warm_window_multiply_adds=macs, dot_product_flops=2 * macs, temporal_memory_bytes=L * 4, excludes='initialization, activation, bias and buffer copies')
    H, L, M = (cfg.hidden, cfg.latent_dim, cfg.decoder_dim)
    gates = 3 if trainer.ARCHITECTURE == 'gru2' else 4
    macs = 48 * L + gates * H * (L + H) + H * M + M * 22 + 48 * 22
    return dict(warm_window_multiply_adds=macs, dot_product_flops=2 * macs, excludes='bias, gate operations, activation, initialization and buffer copies', temporal_memory_bytes=H * (1 if gates == 3 else 2) * 4)

def plot_each_model_feature(runs, tables, names, tag, out, points, no_show, plt):
    left = runs[0]
    count = len(left['prediction'])
    indices = np.unique(np.linspace(0, count - 1, min(count, points), dtype=np.int64))
    targets = left['start'] + indices
    times = left['offset'] + targets * left['dt']
    manifest = []
    for name, run, table in zip(names, runs, tables):
        model_folder = out / tag / 'feature_plots' / name.replace(' ', '_')
        model_folder.mkdir(parents=True, exist_ok=True)
        for j, feature in enumerate(OUTPUT_COLS):
            fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
            manager = fig.canvas.manager
            if hasattr(manager, 'set_window_title'):
                manager.set_window_title(f'{name} | {feature} | {tag}')
            actual_source = scalar_source(run['cache'], STATE_INDICES[j], run['start'], run['mean'][j], run['std'][j])
            predicted_source = scalar_source(run['prediction'], j)
            plot_recording_pair(ax, actual_source, predicted_source, count, run['dt'], run['offset'] + run['start'] * run['dt'], name, points)
            row = table.iloc[j]
            ax.set_title(f'{name} vs actual: {feature}' + chr(10) + f'{tag} | RMSE={row.RMSE:.6g} | MAE={row.MAE:.6g} | 1-R2={1-row.R2:.3e} | NRMSE={row.NRMSE_train_std:.6g}')
            ax.set_xlabel('CSV time [s]')
            ax.set_ylabel(feature)
            ax.legend()
            ax.grid(alpha=0.3)
            path = model_folder / f'feature_{j + 1:02d}.png'
            fig.savefig(path, dpi=130)
            manifest.append(dict(model=name, feature=feature, mode=tag, path=str(path.resolve())))
            if no_show:
                plt.close(fig)
    return manifest

def main():
    p = argparse.ArgumentParser(description='iMotion residual dynamics')
    p.add_argument('--gru-checkpoint', '--gru2-checkpoint', dest='gru_checkpoint', type=Path, help='GRU checkpoint')
    p.add_argument('--lstm-checkpoint', type=Path, help='LSTM checkpoint')
    p.add_argument('--transformer-checkpoint', type=Path, default=ROOT / 'Transformers/artifacts/best_model.pt')
    p.add_argument('--compact-checkpoint', type=Path, default=ROOT / 'Compact Recurrent/artifacts/best_model.pt')
    p.add_argument('--compact-results', type=Path)
    p.add_argument('--no-show', action='store_true', help='Save figures without popup windows')
    p.add_argument('--plot-mode', choices=['measured_one_step', 'recursive', 'mpc', 'all'], help='Popup scenario; default MPC reset if available, otherwise the selected mode')
    p.add_argument('--plot-points', type=int, default=6000, help='Min/max display bins use every sample; zoom for raw detail; metrics use every target row')
    p.add_argument('--transformer-results', type=Path)
    p.add_argument('--gru-results', type=Path)
    p.add_argument('--lstm-results', type=Path)
    p.add_argument('--test-file', default='Simulation_Output_53.csv')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--mode', choices=['reset', 'both', 'measured', 'recursive'], default='reset')
    p.add_argument('--reinit-every', type=int, help='Recursive reset horizon; default is 200 internal prediction samples')
    p.add_argument('--saved-only', action='store_true', help='Load cached results only; no inference')
    p.add_argument('--allow-prefix', action='store_true', help='Permit explicitly limited evaluations; both must cover identical targets')
    a = p.parse_args()
    if a.plot_points < 2:
        p.error('--plot-points must be at least 2')
    if a.gru_checkpoint is None or a.lstm_checkpoint is None:
        default_gru, default_lstm = latest_pair()
    checkpoints = [a.gru_checkpoint or default_gru, a.lstm_checkpoint or default_lstm, a.transformer_checkpoint, a.compact_checkpoint]
    import gru2, lstm4, transformer, trainCompactLatent
    names = ('GRU', 'LSTM', 'Transformer', 'Compact Recurrent')
    trainers = (gru2, lstm4, transformer, trainCompactLatent)
    loaded = []
    for checkpoint, trainer in zip(checkpoints, trainers):
        if not checkpoint.is_file():
            p.error(f'Missing checkpoint: {checkpoint}. Train {trainer.__name__}.py first or supply its checkpoint option.')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if ck.get('format') != trainer.FORMAT:
            p.error(f"Expected updated {getattr(trainer, 'ARCHITECTURE', 'compact')} checkpoint: {checkpoint}")
        history = checkpoint.parent / 'training_history.json'
        expected = ck['training_config']['arguments']['epochs']
        if not history.is_file() or len(json.loads(history.read_text())) < expected:
            p.error(f'Training has not completed: {checkpoint.parent}')
        loaded.append(ck)
    if a.reinit_every is None:
        a.reinit_every = 200
    if a.reinit_every < 1:
        p.error('--reinit-every must be positive')
    stem = Path(a.test_file).stem
    left_root, left_hash = result_root(checkpoints[0], stem, a.gru_results)
    right_root, right_hash = result_root(checkpoints[1], stem, a.lstm_results)
    transformer_root, transformer_hash = result_root(checkpoints[2], stem, a.transformer_results)
    compact_root, compact_hash = result_root(checkpoints[3], stem, a.compact_results, legacy_compact=True)
    roots = [left_root, right_root, transformer_root, compact_root]
    comparison_id = hashlib.sha256('|'.join((left_hash, right_hash, transformer_hash, compact_hash)).encode()).hexdigest()[:10]
    out = (a.output_dir or ROOT / 'results/comparison' / stem / comparison_id / f'r{a.reinit_every}').resolve()
    scenarios = []
    if a.mode in ('both', 'measured'):
        scenarios.append(('measured_one_step', 'measured_one_step'))
    if a.mode == 'reset':
        scenarios.append((f'recursive_reinit{a.reinit_every}', 'recursive'))
    if a.mode in ('both', 'recursive'):
        scenarios += [('recursive', 'recursive'), (f'recursive_reinit{a.reinit_every}', 'recursive')]
    for architecture, root, checkpoint in zip(('gru2', 'lstm4', 'transformer', 'compact'), roots, checkpoints):
        ensure_results(root, checkpoint, architecture, scenarios, a)
    plot_tag = a.plot_mode
    if plot_tag is None:
        plot_tag = f'recursive_reinit{a.reinit_every}' if any((t.startswith('recursive_reinit') for t, _ in scenarios)) else scenarios[0][0]
    if plot_tag == 'mpc':
        plot_tag = f'recursive_reinit{a.reinit_every}'
    if plot_tag != 'all' and plot_tag not in [t for t, _ in scenarios]:
        p.error('Plot mode is not included in selected evaluation modes')
    import matplotlib
    matplotlib.use('Agg' if a.no_show else 'TkAgg')
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.rcParams['figure.max_open_warning'] = 0
    plot_manifest = []
    coverage_rows = []
    temporal_metrics = []
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_timing = []
    summaries = []
    logs = []
    provenance = []

    def emit(message):
        print(message, flush=True)
        logs.append(message)
    metadata = []
    for name, checkpoint, trainer in zip(names, checkpoints, trainers):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if ck.get('format') not in (trainer.FORMAT, getattr(trainer, 'LEGACY_FORMAT', trainer.FORMAT)):
            raise ValueError(f'Wrong checkpoint format for {name}: {checkpoint}')
        model, _ = trainer.load_bundle(checkpoint)
        budget = architecture_budget(trainer, model)
        metadata.append(dict(model=name, compute_budget=budget, checkpoint=str(checkpoint.resolve()), best_epoch=ck['epoch'], parameters=sum((p.numel() for p in model.parameters())), model_config=ck['model_config'], checkpoint_selection=ck.get('training_config', {}).get('checkpoint_selection', 'original training criterion'), training_loss_weights=ck.get('training_config', {}).get('loss_weights', {})))
    controls = gru2.ANGLE_COLS + gru2.VELOCITY_COLS
    emit('GRU vs LSTM vs Transformer vs Compact Recurrent: cached evaluations. Error metrics use physical units per state; NRMSE uses training std.\nTimings are warm CPU benchmarks from saved evaluations; benchmark dates may differ.\nSingle-call time and amortized chunk throughput are separate. R2 is undefined for constant truth.')
    emit(json.dumps(metadata, indent=2))
    keys = ('epochs', 'batch_size', 'steps_per_epoch', 'angle_weight', 'velocity_weight', 'short_horizon', 'medium_horizon', 'long_horizon', 'medium_probability', 'long_probability', 'selection_horizon', 'warmup_epochs', 'learning_rate', 'burn_in_steps', 'grad_block', 'val_horizons', 'val_windows', 'scale_samples_per_file', 'threads', 'seed')
    settings = [ck['training_config']['arguments'] for ck in loaded]
    differences = {key: [value.get(key) for value in settings] for key in keys if any((value.get(key) != settings[0].get(key) for value in settings[1:]))}
    emit('Shared training settings: ' + ('matched' if not differences else json.dumps(differences)))
    if any((manifest_signature(ck['training_config']['training_manifest']) != manifest_signature(loaded[0]['training_config']['training_manifest']) for ck in loaded[1:])):
        raise ValueError('Training recordings or split manifests differ')
    emit('History: GRU/LSTM keep gated memory; Transformer keeps a fixed encoded-token window, initially empty. All scoring starts with two measured rows. Transformer latency uses a fully populated window.')
    for tag, mode in scenarios:
        folders = [root / tag for root in roots]
        runs = [read_result(folder, 'gru', mode) for folder in folders]
        left = runs[0]
        requested_source = (Path(a.test_file) if Path(a.test_file).is_absolute() else RAW_DATA_DIR / 'Test' / a.test_file).resolve()
        if any((Path(run['report']['source_csv']).resolve() != requested_source for run in runs)):
            raise ValueError('Saved evaluations do not match --test-file')
        if any((Path(left['report']['source_csv']).resolve() != Path(run['report']['source_csv']).resolve() for run in runs[1:])):
            raise ValueError('Test recordings differ')
        for run, checkpoint in zip(runs, checkpoints):
            if Path(run['report']['checkpoint']).resolve() != checkpoint.resolve():
                raise ValueError('Checkpoint path differs from saved evaluation')
            expected = a.reinit_every if tag.startswith('recursive_reinit') else 0
            if run['report'].get('reinit_every', 0) != expected:
                raise ValueError('Reset horizons differ')
        if any(((left['start'], len(left['prediction']), left['dt'], left['offset']) != (run['start'], len(run['prediction']), run['dt'], run['offset']) for run in runs[1:])):
            raise ValueError('Saved metrics cover different target intervals; evaluate both over identical rows')
        for key in ('mean', 'std'):
            if any((not np.array_equal(left[key], run[key]) for run in runs[1:])):
                raise ValueError('Scalers differ')
        if any((Path(left['report']['cache']).resolve() != Path(run['report']['cache']).resolve() for run in runs[1:])):
            raise ValueError('Truth cache paths differ; use the same prepared test cache')
        transformer_run = runs[2]
        if transformer_run['report'].get('timing_history_tokens', 0) < loaded[2]['model_config']['history']:
            raise ValueError('Transformer latency was not measured with a full history window')
        for name, run, folder in zip(names, runs, folders):
            coverage = recording_coverage(run['report']['cache'], len(run['prediction']), run['start'], run['dt'], run['offset'], run['report'].get('reinit_every', 0), run['report']['source_csv'])
            coverage.update(model=name, mode=tag)
            coverage_rows.append(coverage)
            rolling = time_metrics(run['prediction'], run['cache'], list(STATE_INDICES), run['mean'], run['std'], run['start'], run['dt'], run['offset'], OUTPUT_COLS)
            rolling.insert(0, 'model', name)
            rolling.insert(0, 'mode', tag)
            temporal_metrics.append(rolling)
            emit(f'FULL-RECORDING COVERAGE: {name}/{tag}: {coverage["prediction_rows"]:,}/{coverage["expected_prediction_rows"]:,} predictions; CSV {coverage["first_prediction_csv_time_s"]:.6f}..{coverage["last_prediction_csv_time_s"]:.6f}s; full={coverage["full_retained_recording"]}; reset blocks={coverage["reset_blocks"]:,}')
        tables = []
        timings = []
        emit(f"\n===== {tag}: {len(left['prediction']):,} targets; retained rows start at {left['start']}; dt={left['dt'] * 1000000.0:g} us =====")
        if any((r['report'].get('max_rows') is not None for r in runs)):
            emit('EXPLICIT PREFIX COMPARISON; not a full-record result')
        for name, folder, run in zip(names, folders, runs):
            table = pd.read_csv(folder / 'metrics.csv')
            table['R2_deficit'] = 1 - table.R2
            table['RMSE_relative_to_persistence'] = table.RMSE / table.persistence_RMSE.replace(0, np.nan)
            if table.feature.tolist() != OUTPUT_COLS:
                raise ValueError('Metric feature order differs')
            if not np.isfinite(table[['RMSE', 'MAE', 'max_abs_error', 'NRMSE_train_std']]).all().all():
                raise ValueError('Nonfinite saved error metrics')
            table.insert(0, 'model', name)
            table.insert(0, 'mode', tag)
            tables.append(table)
            all_metrics.append(table)
            emit(f'\n{name}:\n' + table.to_string(index=False, float_format=lambda v: f'{v:.8g}'))
            report = run['report']
            latency = report['latency']
            throughput = report['throughput']
            timing = dict(mode=tag, model=name, backend='numba', core_precision='float32', history_tokens=report.get('timing_history_tokens'), **latency, inference_seconds=throughput['inference_seconds'], simulated_seconds=throughput['simulated_seconds'], chunk_us_per_prediction=throughput.get('amortized_us_per_step', throughput.get('amortized_us_per_prediction')), chunk_RTF=throughput['RTF'], chunk_rows=throughput['chunk_rows'])
            horizon_latency = report.get('horizon_latency') or {}
            timing.update(reset_horizon_steps=horizon_latency.get('horizon_steps'), reset_horizon_median_us=horizon_latency.get('median_us'), reset_horizon_mean_us=horizon_latency.get('mean_us'), reset_horizon_p95_us=horizon_latency.get('p95_us'))
            timings.append(timing)
            all_timing.append(timing)
            summaries.append(dict(mode=tag, model=name, parameters=metadata[len(timings) - 1]['parameters'], warm_window_MACs=metadata[len(timings) - 1]['compute_budget']['warm_window_multiply_adds'], temporal_memory_bytes=metadata[len(timings) - 1]['compute_budget']['temporal_memory_bytes'], mean_NRMSE_train_std=table.NRMSE_train_std.mean(), control_mean_NRMSE=table.loc[table.feature.isin(controls), 'NRMSE_train_std'].mean(), control_rms_NRMSE=np.sqrt(np.mean(table.loc[table.feature.isin(controls), 'NRMSE_train_std'] ** 2)), rms_NRMSE_train_std=np.sqrt(np.mean(table.NRMSE_train_std ** 2)), median_call_us=latency['median_us'], mean_call_us=latency['mean_us'], p95_call_us=latency['p95_us'], chunk_us_per_prediction=timing['chunk_us_per_prediction'], chunk_RTF=timing['chunk_RTF']))
            summaries[-1].update(reset_horizon_median_us=horizon_latency.get('median_us'), reset_horizon_mean_us=horizon_latency.get('mean_us'), reset_horizon_p95_us=horizon_latency.get('p95_us'))
        control_table = pd.concat(tables).loc[lambda table: table.feature.isin(controls)]
        emit('\nMPC state metrics (angles and angular velocities):\n' + control_table.to_string(index=False, float_format=lambda v: f'{v:.8g}'))
        emit('\nAll saved timing metrics:\n' + pd.DataFrame(timings).to_string(index=False))
        delta = pd.DataFrame({'feature': OUTPUT_COLS})
        for metric in ('RMSE', 'MAE', 'R2', 'max_abs_error', 'NRMSE_train_std'):
            for name, table in zip(names, tables):
                delta[f'{metric}_{name}'] = table[metric].to_numpy()
            for name, table in zip(names[1:], tables[1:]):
                delta[f'{metric}_change_{name}_minus_GRU'] = table[metric].to_numpy() - tables[0][metric].to_numpy()
        folder = out / tag
        folder.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).to_csv(folder / 'metrics.csv', index=False)
        pd.DataFrame(timings).to_csv(folder / 'timing.csv', index=False)
        delta.to_csv(folder / 'metric_differences.csv', index=False)
        control_table.to_csv(folder / 'control_metrics.csv', index=False)
        if plot_tag == 'all' or plot_tag == tag:
            plot_temporal_errors(pd.concat(temporal_metrics, ignore_index=True), names, tag, out, plt, a.no_show)
            plot_manifest.extend(plot_each_model_feature(runs, tables, names, tag, out, a.plot_points, a.no_show, plt))
        provenance.append(dict(mode=tag, target_start=left['start'], predictions=len(left['prediction']), results=[str(f.resolve()) for f in folders]))
    pd.DataFrame(coverage_rows).to_csv(out / 'recording_coverage.csv', index=False)
    pd.concat(temporal_metrics, ignore_index=True).to_csv(out / 'error_over_time.csv', index=False)
    pd.concat(all_metrics).to_csv(out / 'all_metrics.csv', index=False)
    pd.DataFrame(all_timing).to_csv(out / 'all_timing.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'model_summary.csv', index=False)
    emit('\nSUMMARY (lower NRMSE and time is better):\n' + summary.to_string(index=False, float_format=lambda v: f'{v:.8g}'))
    (out / 'provenance.json').write_text(json.dumps(dict(models=metadata, scenarios=provenance), indent=2), encoding='utf-8')
    tags = summary['mode'].unique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, key, label in zip(axes, ('mean_NRMSE_train_std', 'control_mean_NRMSE', 'median_call_us'), ('Mean normalized RMSE, all states', 'Mean normalized RMSE, MPC states', 'Median physical call [us]')):
        view = summary.pivot(index='mode', columns='model', values=key).reindex(tags)
        view.plot.bar(ax=ax, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'], rot=15)
        ax.set_ylabel(label)
        ax.set_xlabel('')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.25)
    fig.savefig(out / 'accuracy_timing_comparison.png', dpi=150)
    plt.close(fig)
    emit(f'\nSaved direct comparison: {out}')
    (out / 'comparison.txt').write_text('\n'.join(logs), encoding='utf-8')
    (out / 'feature_plot_manifest.json').write_text(json.dumps(plot_manifest, indent=2), encoding='utf-8')
    if not a.no_show:
        emit(f'Opened {len(plot_manifest)} separate model/feature figures. Close ALL windows to finish.')
        plt.show(block=True)
    plt.close('all')
if __name__ == '__main__':
    main()
