import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, resolve_array, same_recording
import argparse
import time
import platform
import numba
import numpy as np
import pandas as pd
from common.recording_view import recording_coverage, save_coverage, scalar_source, plot_recording_pair, time_metrics
import hashlib
import importlib
selector = argparse.ArgumentParser(add_help=False)
selector.add_argument('--architecture', choices=['gru2', 'lstm4', 'transformer', 'compact'], required=True)
selection, _ = selector.parse_known_args()
trainer = importlib.import_module({'gru2': 'gru2', 'lstm4': 'lstm4', 'transformer': 'transformer', 'compact': 'trainCompactLatent'}[selection.architecture])
for symbol in ('SCRIPT_DIR', 'DEFAULT_DATASET_DIR', 'DEFAULT_ARTIFACT_DIR', 'STATE_INDICES', 'RESULT_DIR', 'OUTPUT_COLS', 'load_bundle', 'export_weights', 'native_encode', 'native_chunk', 'physical_step', 'write_json'):
    globals()[symbol] = getattr(trainer, symbol)
MODEL_NAME = {'gru2': 'GRU2', 'lstm4': 'LSTM4', 'transformer': 'Transformer', 'compact': 'Compact Recurrent'}[selection.architecture]
from common.evaluation import Metrics, load_test

def main():
    p = argparse.ArgumentParser(description='iMotion residual dynamics')
    p.add_argument('--architecture', choices=['gru2', 'lstm4', 'transformer', 'compact'], required=True)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_ARTIFACT_DIR / 'best_model.pt')
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument('--test-file', type=Path, default=Path('Simulation_Output_53.csv'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--mode', choices=['both', 'measured', 'recursive'], default='both')
    p.add_argument('--recursive', action='store_true', help='Recursive-only alias')
    p.add_argument('--chunk-rows', type=int, default=20000)
    p.add_argument('--reinit-every', type=int, default=0, help='Reset recursive latent state from measured rows every N predictions; 0 keeps continuous rollout')
    p.add_argument('--latency-repeats', type=int, default=3000)
    p.add_argument('--max-rows', type=int, help='Explicit prefix limit; default entire retained CSV')
    p.add_argument('--show', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--plot-points', type=int, default=10000)
    a = p.parse_args()
    if min(a.chunk_rows, a.latency_repeats) < 1 or a.reinit_every < 0 or (a.max_rows is not None and a.max_rows < 3):
        p.error('Invalid sizes')
    if a.recursive:
        if a.mode == 'measured':
            p.error('Conflicting mode flags')
        a.mode = 'recursive'
    model, ck = load_bundle(a.checkpoint)
    sc = ck['scaler']
    w = export_weights(model)
    source = (a.test_file if a.test_file.is_absolute() else RAW_DATA_DIR / 'Test' / a.test_file).resolve()
    if any((same_recording(e['source_csv'], source) for e in ck['training_config']['training_manifest'])):
        p.error('Choose a held-out Test CSV')
    if any((same_recording(f, source) for f in sc.get('fitted_files', []))):
        p.error('CSV contributed to scaler fitting')
    base = a.output_dir or RESULT_DIR / source.stem / f'checkpoint_{hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()[:16]}'
    base.mkdir(parents=True, exist_ok=True)
    x, cache = load_test(source, a.dataset_dir, sc, base)
    if a.max_rows is not None:
        x = x[:a.max_rows]
    if len(x) < 3:
        raise ValueError('At least three retained rows required')
    mean = np.asarray(sc['mean'], np.float64)
    std = np.asarray(sc['std'], np.float64)
    dt = float(sc['sampling_interval_seconds'])
    offset = int(sc['startup_rows_removed']) * dt
    n = len(x) - 2

    def physical(y):
        return np.asarray(y, dtype=np.float64) * std[list(STATE_INDICES)] + mean[list(STATE_INDICES)]
    prev = np.array(x[0], copy=True)
    current = np.array(x[1], copy=True)
    z = native_encode(current, w[0], w[1])
    physical_prev = prev.astype(float) * std + mean
    physical_current = current.astype(float) * std + mean
    timing_z = z.copy()
    timing_history_tokens = None
    if a.architecture == 'transformer':
        prefix = min(model.config.history, len(x) - 2)
        _, _, _, timing_z = native_chunk(x[1:prefix + 1], prev, current, z, False, mean, std, *w)
        physical_prev = x[prefix].astype(float) * std + mean
        physical_current = x[prefix + 1].astype(float) * std + mean
        timing_history_tokens = prefix
    print('Compiling/warming Numba kernels (excluded from latency)...', flush=True)
    for recursive in (False, True):
        native_chunk(x[1:3], prev, current, z, recursive, mean, std, *w)
    for _ in range(30):
        physical_step(physical_prev, physical_current, timing_z, mean, std, *w)
    import matplotlib
    if not a.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = ['measured_one_step', 'recursive'] if a.mode == 'both' else ['recursive' if a.mode == 'recursive' else 'measured_one_step']
    if a.reinit_every and 'recursive' not in modes:
        p.error('--reinit-every applies only to recursive mode')
    summaries = []
    for mode in modes:
        reinit = a.reinit_every if mode == 'recursive' else 0
        tag = mode + (f'_reinit{reinit}' if reinit else '')
        out = base / tag
        out.mkdir(parents=True, exist_ok=True)
        log = (out / 'results.txt').open('w', encoding='utf-8')

        def emit(v):
            print(v, flush=True)
            print(v, file=log, flush=True)
        emit(f"\n===== {mode} | {MODEL_NAME} Numba | epoch {ck['epoch']} =====")
        emit(f'{n:,} predictions; two measured initialization rows; CSV times {offset + 2 * dt:g}..{offset + (len(x) - 1) * dt:g}s; dt={dt * 1000000.0:g}us.')
        if reinit:
            emit(f'Recursive latent state resets from measured rows every {reinit} predictions.')
        emit('Prepared inputs are normalized; current u[t] predicts x[t+1]. ' + ('Latent state resets from measured rows at each chunk.' if reinit else 'Latent state persists across chunks.') + ' Centered 51-row command smoothing makes this an offline test, not verified causal control.')
        if a.max_rows:
            emit(f'EXPLICIT PREFIX: {len(x)} retained rows')
        times = []
        for _ in range(a.latency_repeats):
            start = time.perf_counter_ns()
            physical_step(physical_prev, physical_current, timing_z, mean, std, *w)
            times.append((time.perf_counter_ns() - start) / 1000)
        latency = dict(mean_us=float(np.mean(times)), median_us=float(np.median(times)), p95_us=float(np.percentile(times, 95)), max_us=float(np.max(times)), deadline_us=dt * 1000000.0, deadline_miss_percent=float(np.mean(np.array(times) > dt * 1000000.0) * 100), RTF=float(np.mean(times) / (dt * 1000000.0)), repeats=a.latency_repeats)
        emit(f'Warm physical one-step call (includes normalization/reconstruction; latent already initialized): {latency}')
        previous = prev.copy()
        cur = current.copy()
        latent = z.copy()
        prediction = np.lib.format.open_memmap(out / 'predictions_physical.npy', mode='w+', dtype=np.float64, shape=(n, 22))
        meter = Metrics(std[list(STATE_INDICES)])
        seconds = 0.0
        wall = time.perf_counter()
        last = wall
        chunk_size = reinit or a.chunk_rows
        for begin in range(0, n, chunk_size):
            end = min(n, begin + chunk_size)
            started = time.perf_counter()
            if mode == 'recursive' and reinit:
                previous = np.array(x[begin], copy=True)
                cur = np.array(x[begin + 1], copy=True)
                latent = native_encode(cur, w[0], w[1])
            result, previous, cur, latent = native_chunk(x[begin + 1:end + 1], previous, cur, latent, mode == 'recursive', mean, std, *w)
            seconds += time.perf_counter() - started
            if not np.isfinite(result).all():
                raise RuntimeError(f'{mode} diverged after prediction {begin}')
            prediction[begin:end] = result
            actual = physical(x[begin + 2:end + 2, list(STATE_INDICES)])
            baseline_row = begin + 1 if reinit else 1
            baseline = np.broadcast_to(physical(x[baseline_row, list(STATE_INDICES)]), actual.shape) if mode == 'recursive' else physical(x[begin + 1:end + 1, list(STATE_INDICES)])
            meter.add(actual, result, baseline)
            now = time.perf_counter()
            if now - last > 5 or end == n:
                emit(f'{end:,}/{n:,} predictions; elapsed={now - wall:.1f}s; remaining={(now - wall) * (n - end) / end:.1f}s')
                last = now
        prediction.flush()
        metrics = meter.table()
        metrics.to_csv(out / 'metrics.csv', index=False)
        throughput = dict(inference_seconds=seconds, simulated_seconds=n * dt, amortized_us_per_step=seconds / n * 1000000.0, RTF=seconds / (n * dt), chunk_rows=chunk_size, reinit_every=reinit)
        emit('\n' + metrics.to_string(index=False, float_format=lambda v: f'{v:.7g}'))
        emit(f'Compiled chunk throughput (NOT single-call latency): {throughput}')
        write_json(out / 'evaluation.json', dict(mode=mode, tag=tag, checkpoint=str(a.checkpoint.resolve()), source_csv=str(source), cache=str(cache), architecture=a.architecture, epoch=ck['epoch'], timing_history_tokens=timing_history_tokens, inference_backend='numba', core_precision='float32', benchmark_environment=dict(cpu=platform.processor(), machine=platform.machine(), python=platform.python_version(), numba=numba.__version__, numpy=np.__version__, torch=str(trainer.torch.__version__)), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(), predictions=n, initial_measured_rows=2, dt_s=dt, excluded_seconds=offset, reinit_every=reinit, latent_resets=int(np.ceil(n / reinit)) if reinit else 0, latency=latency, throughput=throughput, max_rows=a.max_rows, timing_excludes='JIT compilation, initialization, CSV/cache I/O, file writes, scoring and plotting', command_smoothing='centered Savitzky-Golay 51 rows, offline'))
        coverage = recording_coverage(cache, n, 2, dt, offset, reinit, source)
        save_coverage(out / 'recording_coverage.json', coverage)
        emit(f'Coverage: {n:,}/{coverage["expected_prediction_rows"]:,} predictions; full retained CSV={coverage["full_retained_recording"]}')
        time_metrics(prediction, x, list(STATE_INDICES), mean[list(STATE_INDICES)], std[list(STATE_INDICES)], 2, dt, offset, OUTPUT_COLS).to_csv(out / 'error_over_time.csv', index=False)
        indices = np.unique(np.linspace(0, n - 1, min(n, a.plot_points), dtype=int))
        t = offset + (indices + 2) * dt
        actual = physical(x[indices + 2][:, list(STATE_INDICES)])
        forecast = prediction[indices]
        for page in range(2):
            fig, axes = plt.subplots(6, 2, figsize=(16, 15), sharex=True, layout='constrained')
            for slot, ax in enumerate(axes.flat):
                if slot == 11:
                    ax.set_visible(False)
                    continue
                j = page * 11 + slot
                plot_recording_pair(ax, scalar_source(x, STATE_INDICES[j], 2, mean[STATE_INDICES[j]], std[STATE_INDICES[j]]), scalar_source(prediction, j), n, dt, offset + 2 * dt, MODEL_NAME, a.plot_points)
                ax.set_title(f'{OUTPUT_COLS[j]} | RMSE={metrics.RMSE.iloc[j]:.5g}')
                ax.set_xlabel('CSV time [s]')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            fig.suptitle(f'{source.name}: {tag}; {n:,} predictions across the full selected CSV\nAll samples contribute to min/max bands; zoom/pan for detail')
            fig.savefig(out / f'actual_vs_predicted_{page + 1}.png', dpi=130)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        axes[0].bar(['Mean', 'p95', 'Max', 'Deadline'], [latency['mean_us'], latency['p95_us'], latency['max_us'], latency['deadline_us']])
        axes[0].set_ylabel('One-step microseconds')
        axes[1].bar(['One-step call', 'Chunk throughput'], [latency['RTF'], throughput['RTF']])
        axes[1].axhline(1, color='red', linestyle='--')
        axes[1].set_ylabel('RTF')
        axes[1].set_yscale('log')
        fig.suptitle(mode)
        fig.savefig(out / 'latency.png', dpi=150)
        summaries.append(dict(mode=mode, predictions=n, mean_R2=metrics.R2.mean(), mean_NRMSE=metrics.NRMSE_train_std.mean(), mean_call_us=latency['mean_us'], p95_call_us=latency['p95_us'], throughput_RTF=throughput['RTF']))
        emit(f'Results: {out}')
        log.close()
        if not a.show:
            plt.close('all')
    pd.DataFrame(summaries).to_csv(base / 'method_comparison.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    if a.show:
        plt.show(block=True)
    plt.close('all')
if __name__ == '__main__':
    main()
