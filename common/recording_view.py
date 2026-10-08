import json
from pathlib import Path

import numpy as np
import pandas as pd


def recording_coverage(cache, count, start, dt, offset, reset=0, source=None):
    metadata_path = Path(cache).with_suffix('.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    if 'rows' in metadata:
        retained = int(metadata['rows'])
    else:
        retained = len(np.load(cache, mmap_mode='r'))
    trim = int(round(offset / dt))
    return dict(source_csv=str(source) if source else metadata.get('source'),
                raw_rows=retained + trim, startup_rows_excluded=trim, retained_rows=retained,
                initialization_rows=start, prediction_rows=count,
                expected_prediction_rows=retained - start,
                full_retained_recording=count == retained - start,
                first_prediction_csv_time_s=offset + start * dt,
                last_prediction_csv_time_s=offset + (start + count - 1) * dt,
                simulated_prediction_duration_s=count * dt,
                reset_every_samples=reset, reset_horizon_s=reset * dt,
                reset_blocks=int(np.ceil(count / reset)) if reset else 0,
                tail_prediction_rows=0 if count == retained - start else retained - start - count)


def save_coverage(path, coverage):
    Path(path).write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    return coverage


def scalar_source(array, column, start=0, mean=0., std=1.):
    def read(begin, end):
        values = np.asarray(array[start + begin:start + end, column], dtype=float)
        return values * std + mean if std != 1. or mean != 0. else values
    return read


def envelope(read, begin, end, budget):
    count = end - begin
    if count < 1:
        return (np.empty(0),) * 4
    values = read(begin, end)
    if count <= budget:
        return np.arange(begin, end, dtype=float), values, values, values
    edges = np.linspace(0, count, min(count, budget) + 1, dtype=np.int64)
    lengths = np.diff(edges)
    minimum = np.minimum.reduceat(values, edges[:-1])
    maximum = np.maximum.reduceat(values, edges[:-1])
    average = np.add.reduceat(values, edges[:-1]) / lengths
    centers = begin + (edges[:-1] + edges[1:] - 1) / 2
    return centers, average, minimum, maximum


class RecordingTrace:
    def __init__(self, ax, read, count, dt, first_time, label, color, budget=6000, linestyle='-'):
        self.ax = ax
        self.read = read
        self.count = count
        self.dt = dt
        self.first = first_time
        self.budget = budget
        self.color = color
        self.line, = ax.plot([], [], color=color, linestyle=linestyle, linewidth=1, label=label)
        self.band = None
        self.bounds = None
        self.busy = False
        self.update(0, count)
        self.callback = ax.callbacks.connect('xlim_changed', self.on_zoom)

    def update(self, begin, end):
        if self.bounds == (begin, end):
            return
        positions, average, minimum, maximum = envelope(self.read, begin, end, self.budget)
        times = self.first + positions * self.dt
        self.line.set_data(times, average)
        if self.band is not None:
            self.band.remove()
        self.band = self.ax.fill_between(times, minimum, maximum, color=self.color, alpha=.15, linewidth=0)
        self.bounds = (begin, end)

    def on_zoom(self, ax):
        if self.busy:
            return
        lower, upper = sorted(ax.get_xlim())
        begin = max(0, int(np.floor((lower - self.first) / self.dt)))
        end = min(self.count, int(np.ceil((upper - self.first) / self.dt)) + 1)
        if end <= begin:
            return
        self.busy = True
        try:
            self.update(begin, end)
            ax.figure.canvas.draw_idle()
        finally:
            self.busy = False


def plot_recording_pair(ax, actual, predicted, count, dt, first_time, name, budget=6000):
    traces = [RecordingTrace(ax, actual, count, dt, first_time, 'Actual: mean and full min/max envelope', 'black', budget),
              RecordingTrace(ax, predicted, count, dt, first_time, name + ': mean and full min/max envelope', 'tab:blue', budget, '--')]
    if not hasattr(ax.figure, '_recording_traces'):
        ax.figure._recording_traces = []
    ax.figure._recording_traces.extend(traces)
    ax.set_xlim(first_time, first_time + max(count - 1, 1) * dt)
    ax.relim()
    limits = [(float(np.min(trace.band.get_paths()[0].vertices[:, 1])), float(np.max(trace.band.get_paths()[0].vertices[:, 1]))) for trace in traces if trace.band.get_paths()]
    if limits:
        lower = min(pair[0] for pair in limits)
        upper = max(pair[1] for pair in limits)
        margin = max((upper - lower) * .04, abs(upper) * 1e-8, 1e-12)
        ax.set_ylim(lower - margin, upper + margin)
    ax.set_xlabel('CSV time [s] — zoom/pan to inspect the underlying samples')


def time_metrics(prediction, truth_array, columns, mean, std, start, dt, offset, features, seconds=1., block=100000):
    width = max(1, int(round(seconds / dt)))
    results = []
    for begin in range(0, len(prediction), width):
        end = min(begin + width, len(prediction))
        count = end - begin
        seen = 0
        average = np.zeros(len(features))
        m2 = np.zeros(len(features))
        sae = np.zeros(len(features))
        sse = np.zeros(len(features))
        maximum = np.zeros(len(features))
        for lo in range(begin, end, block):
            hi = min(lo + block, end)
            actual = np.asarray(truth_array[start + lo:start + hi][:, columns], float) * std + mean
            forecast = np.asarray(prediction[lo:hi], float)
            error = forecast - actual
            size = len(actual)
            local_mean = actual.mean(0)
            delta = local_mean - average
            m2 += ((actual - local_mean) ** 2).sum(0) + delta ** 2 * seen * size / (seen + size)
            average += delta * size / (seen + size)
            seen += size
            sae += abs(error).sum(0)
            sse += (error ** 2).sum(0)
            maximum = np.maximum(maximum, abs(error).max(0))
        variance = m2 / count
        ratio = np.full(len(features), np.nan)
        threshold = np.finfo(float).eps * np.maximum(1., average ** 2 + variance)
        np.divide(sse / count, variance, out=ratio, where=variance > threshold)
        rmse = np.sqrt(sse / count)
        for j, feature in enumerate(features):
            results.append(dict(feature=feature, start_csv_time_s=offset + (start + begin) * dt,
                                end_csv_time_s=offset + (start + end - 1) * dt, predictions=count,
                                MAE=sae[j] / count, RMSE=rmse[j], R2=1 - ratio[j], max_abs_error=maximum[j], NRMSE_train_std=rmse[j] / std[j]))
    return pd.DataFrame(results)


def plot_temporal_errors(table, names, mode, out, plt, no_show):
    features = ['thetalift[rad]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
    view = table[table['mode'] == mode]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, layout='constrained')
    for ax, feature in zip(axes, features):
        for name in names:
            rows = view[(view.model == name) & (view.feature == feature)]
            times = (rows.start_csv_time_s + rows.end_csv_time_s) / 2
            ax.plot(times, rows.RMSE, label=name)
        ax.set_ylabel('RMSE')
        ax.set_title(feature)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('CSV time [s]')
    fig.suptitle(f'{mode}: error over the entire recording; every sample scored in one-second windows')
    fig.savefig(Path(out) / mode / 'error_over_time.png', dpi=130)
    if no_show:
        plt.close(fig)
