import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, resolve_array, same_recording
import argparse
from datetime import datetime, timezone
import hashlib
import time
import json
import platform
import numpy as np
import pandas as pd
from mechanical_compact import SCRIPT_DIR, DEFAULT_DATASET_DIR, DEFAULT_ARTIFACT_DIR, INPUT_COLS, OUTPUT_COLS, STATE_INDICES, COMMAND_INDICES, load_bundle, load_test_rows, export_weights, native_encode, native_chunk, physical_step, held_command, Metrics, write_json

def benchmark(fn, repeats, deadline_us):
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1000)
    return dict(mean_us=float(np.mean(times)), median_us=float(np.median(times)), p95_us=float(np.percentile(times, 95)), max_us=float(np.max(times)), repeats=repeats, deadline_us=deadline_us, deadline_miss_percent=float(np.mean(np.asarray(times) > deadline_us) * 100), RTF=float(np.mean(times) / deadline_us))

def main():
    p = argparse.ArgumentParser(description='iMotion residual dynamics')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_ARTIFACT_DIR / 'best_model.pt')
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument('--test-file', type=Path, default=Path('Simulation_Output_53.csv'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--mode', choices=['both', 'measured', 'recursive'], default='both')
    p.add_argument('--mpc-horizons', type=int, nargs='+', default=[100, 500, 2000], help='Reset horizons in internal samples: 1/5/20ms at 10us')
    p.add_argument('--skip-continuous', action='store_true', help='Evaluate one-step and MPC reset horizons without the long autonomous rollout')
    p.add_argument('--chunk-rows', type=int, default=20000)
    p.add_argument('--latency-repeats', type=int, default=3000)
    p.add_argument('--control-block', type=int, default=100, help='Held-command timing block; 100 internal steps is 1ms')
    p.add_argument('--max-rows', type=int, help='Explicit limited-prefix test; default entire CSV')
    p.add_argument('--plot-points', type=int, default=6000)
    p.add_argument('--no-show', action='store_true')
    a = p.parse_args()
    if min(a.chunk_rows, a.latency_repeats, a.control_block, a.plot_points, *a.mpc_horizons) < 1 or (a.max_rows is not None and a.max_rows < 3):
        p.error('Positive counts and at least three rows are required')
    if not a.checkpoint.is_file():
        p.error('Checkpoint missing; run trainMechanicalCompact.py first or supply --checkpoint')
    model, ck = load_bundle(a.checkpoint)
    sc = ck['scaler']
    w = export_weights(model)
    mean = np.asarray(sc['mean'], float)
    std = np.asarray(sc['std'], float)
    dt = float(sc['sampling_interval_seconds'])
    offset = sc['startup_rows_removed'] * dt
    source = (a.test_file if a.test_file.is_absolute() else RAW_DATA_DIR / 'Test' / a.test_file).resolve()
    if any((same_recording(entry['source_csv'], source) for entry in ck['training_config']['training_manifest'])):
        p.error('Choose a held-out test recording')
    if any((same_recording(path, source) for path in sc.get('fitted_files', []))):
        p.error('CSV contributed to training normalization')
    fingerprint = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    base = (a.output_dir or ROOT / 'MPC offline Model/results/plant_evaluation' / source.stem / f'checkpoint_{fingerprint[:16]}' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    if base.exists() and any(base.iterdir()):
        p.error('Choose a fresh output folder')
    base.mkdir(parents=True, exist_ok=True)
    x, cache = load_test_rows(source, a.dataset_dir, sc, base)
    length = min(len(x), a.max_rows) if a.max_rows is not None else len(x)
    if length < 3:
        raise ValueError('Need two initialization rows and one target')
    n = length - 2
    previous = x[0]
    current = x[1]
    z = native_encode(current, w[0], w[1])
    pp = previous.astype(float) * std + mean
    cp = current.astype(float) * std + mean
    print(f"Mechanical-only compact: {sum((p.numel() for p in model.parameters())):,} parameters; 12 features -> 4 residuals; epoch={ck['epoch']}", flush=True)
    print(f'{n:,} predictions; dt={dt * 1000000.0:g}us; no pressure/electrical model inputs', flush=True)
    print('Warming compiled kernels; compilation excluded from timing...', flush=True)
    for mode in (False, True):
        native_chunk(x[1:3], previous, current, z, mode, mean, std, *w)
    for _ in range(30):
        physical_step(pp, cp, z, mean, std, *w)
    held_command(previous, current, z, cp[4:6], a.control_block, mean, std, *w)
    latency = benchmark(lambda: physical_step(pp, cp, z, mean, std, *w), a.latency_repeats, dt * 1000000.0)
    block_latency = benchmark(lambda: held_command(previous, current, z, cp[4:6], a.control_block, mean, std, *w), min(a.latency_repeats, 200), a.control_block * dt * 1000000.0)
    print(f'Physical one-step latency: {latency}', flush=True)
    print(f'Held {a.control_block}-step plant advance: {block_latency}', flush=True)
    scenarios = []
    if a.mode in ('both', 'measured'):
        scenarios.append(('measured_one_step', False, 0))
    if a.mode in ('both', 'recursive'):
        scenarios += [(f'recursive_reinit{h}', True, h) for h in sorted(set(a.mpc_horizons))]
        if not a.skip_continuous:
            scenarios.append(('recursive', True, 0))
    import matplotlib
    matplotlib.use('Agg' if a.no_show else 'TkAgg')
    import matplotlib.pyplot as plt
    plt.ioff()
    summaries = []
    all_metrics = []

    def physical(values):
        return np.asarray(values, float) * std[:4] + mean[:4]

    def emit(message, file):
        print(message, flush=True)
        print(message, file=file, flush=True)
    for tag, recursive, reinit in scenarios:
        out = base / tag
        out.mkdir()
        with (out / 'results.txt').open('w', encoding='utf-8') as log:
            emit(f'=== {tag}: {n:,} predictions ===', log)
            emit('Two measured six-signal rows initialize the model; then only commands enter recursive predictions.', log)
            if reinit:
                emit(f'Explicit measured reinitialization every {reinit} samples ({reinit * dt * 1000:g}ms).', log)
            if a.max_rows is not None:
                emit(f'EXPLICIT PREFIX: {length:,} retained rows.', log)
            pred = np.lib.format.open_memmap(out / 'predictions_physical.npy', mode='w+', dtype=np.float64, shape=(n, 4))
            prev = previous.copy()
            cur = current.copy()
            latent = z.copy()
            meter = Metrics(std[:4])
            seconds = 0.0
            wall = time.perf_counter()
            last = wall
            size = reinit or a.chunk_rows
            for begin in range(0, n, size):
                end = min(begin + size, n)
                tick = time.perf_counter()
                if recursive and reinit:
                    prev = x[begin]
                    cur = x[begin + 1]
                    latent = native_encode(cur, w[0], w[1])
                rows = x[begin + 1:end + 1]
                result, prev, cur, latent = native_chunk(rows, prev, cur, latent, recursive, mean, std, *w)
                seconds += time.perf_counter() - tick
                if not np.isfinite(result).all():
                    raise RuntimeError(f'{tag} diverged at prediction {begin}; incomplete output is not a valid evaluation')
                pred[begin:end] = result
                truth = physical(x[begin + 2:end + 2, :4])
                if recursive:
                    baseline = np.broadcast_to(physical(x[begin + 1 if reinit else 1, :4]), truth.shape)
                else:
                    baseline = physical(x[begin + 1:end + 1, :4])
                meter.add(truth, result, baseline)
                now = time.perf_counter()
                if now - last > 5 or end == n:
                    emit(f'{end:,}/{n:,}; elapsed={now - wall:.1f}s; remaining={(now - wall) * (n - end) / end:.1f}s', log)
                    last = now
            pred.flush()
            metrics = meter.table()
            metrics.to_csv(out / 'metrics.csv', index=False)
            combined = metrics.copy()
            combined.insert(0, 'mode', tag)
            all_metrics.append(combined)
            throughput = dict(inference_seconds=seconds, simulated_seconds=n * dt, amortized_us_per_step=seconds / n * 1000000.0, RTF=seconds / (n * dt), chunk_rows=size, note='includes input projection/copy and initialization; excludes scoring, plotting and prediction file writes')
            emit(metrics.to_string(index=False, float_format=lambda value: f'{value:.8g}'), log)
            emit(f'One-step benchmark: {latency}', log)
            emit(f'Held-command advance benchmark: {block_latency}', log)
            emit(f'Throughput: {throughput}', log)
            write_json(out / 'evaluation.json', dict(mode='recursive' if recursive else 'measured_one_step', tag=tag, checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=fingerprint, source_csv=str(source), cache=str(cache), source_projection_indices=sc['source_projection_indices'], input_columns=INPUT_COLS, output_columns=OUTPUT_COLS, predictions=n, initial_measured_rows=2, reinit_every=reinit, dt_s=dt, excluded_seconds=offset, max_rows=a.max_rows, latency=latency, held_block_latency=block_latency, control_block=a.control_block, throughput=throughput, inference_backend='numba', core_precision='float32', cpu=platform.processor(), command_smoothing='centered 51 rows, offline', excludes_hydraulic_electrical_features=True))
            indices = np.unique(np.linspace(0, n - 1, min(n, a.plot_points), dtype=np.int64))
            times = offset + (indices + 2) * dt
            truth = physical(x[indices + 2, :4])
            forecast = pred[indices]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout='constrained')
            for j, ax in enumerate(axes.flat):
                ax.plot(times, truth[:, j], color='black', label='Recorded')
                ax.plot(times, forecast[:, j], '--', label='Mechanical Compact')
                ax.set_title(f'{OUTPUT_COLS[j]} | RMSE={metrics.RMSE.iloc[j]:.5g}')
                ax.set_ylabel('rad' if j < 2 else 'rad/s')
                ax.set_xlabel('CSV time [s]')
                ax.grid(alpha=0.25)
                ax.legend()
            fig.suptitle(tag)
            fig.savefig(out / 'actual_vs_predicted.png', dpi=150)
            if a.no_show:
                plt.close(fig)
            summaries.append(dict(mode=tag, predictions=n, mean_NRMSE=float(metrics.NRMSE_train_std.mean()), mean_R2=float(metrics.R2.mean()) if metrics.R2.notna().any() else None, mean_call_us=latency['mean_us'], p95_call_us=latency['p95_us'], held_block_mean_us=block_latency['mean_us'], throughput_RTF=throughput['RTF']))
    summary = pd.DataFrame(summaries)
    summary.to_csv(base / 'method_comparison.csv', index=False)
    pd.concat(all_metrics).to_csv(base / 'all_metrics.csv', index=False)
    print('All modes:' + chr(10) + summary.to_string(index=False, float_format=lambda v: f'{v:.8g}'), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axes[0].bar(['one-step mean', 'one-step p95', 'held block mean', 'held block p95'], [latency['mean_us'], latency['p95_us'], block_latency['mean_us'], block_latency['p95_us']])
    axes[0].set_ylabel('Physical call latency [us]')
    axes[0].set_yscale('log')
    axes[0].tick_params(axis='x', rotation=15)
    axes[1].bar(summary['mode'], summary['throughput_RTF'])
    axes[1].axhline(1, color='red', linestyle='--')
    axes[1].set_ylabel('Inference throughput RTF')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=15)
    fig.savefig(base / 'timing_summary.png', dpi=150)
    print(f'Saved mechanical-model evaluation: {base}', flush=True)
    if not a.no_show:
        print('Close ALL plot windows to finish.', flush=True)
        plt.show(block=True)
    plt.close('all')
if __name__ == '__main__':
    main()
