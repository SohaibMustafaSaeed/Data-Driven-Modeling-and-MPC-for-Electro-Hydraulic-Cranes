import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from mechanical_compact import native_chunk as plant_chunk
from crane_kinematics import read_cartesian_reference
try:
    from common.recording_view import plot_recording_pair, scalar_source
except ModuleNotFoundError:
    from recording_view import plot_recording_pair, scalar_source


class StreamMetrics:
    def __init__(self, names):
        self.names = names
        self.count = 0
        self.mean = np.zeros(len(names))
        self.m2 = self.mean.copy()
        self.sse = self.mean.copy()
        self.sae = self.mean.copy()
        self.maximum = self.mean.copy()

    def add(self, reference, actual):
        size = len(reference)
        mean = reference.mean(0)
        delta = mean - self.mean
        self.m2 += ((reference - mean) ** 2).sum(0) + delta ** 2 * self.count * size / (self.count + size)
        self.mean += delta * size / (self.count + size)
        self.count += size
        error = actual - reference
        self.sse += (error ** 2).sum(0)
        self.sae += abs(error).sum(0)
        self.maximum = np.maximum(self.maximum, abs(error).max(0))

    def table(self):
        variance = self.m2 / self.count
        ratio = np.full(len(self.names), np.nan)
        threshold = np.finfo(float).eps * np.maximum(1., self.mean ** 2 + variance)
        np.divide(self.sse / self.count, variance, out=ratio, where=variance > threshold)
        return pd.DataFrame(dict(feature=self.names, RMSE=np.sqrt(self.sse / self.count), MAE=self.sae / self.count, max_abs_error=self.maximum, R2=1 - ratio))


def kernels(trainer, state_indices):
    step = trainer.native_step
    si = np.asarray(state_indices, np.int64)

    @njit(cache=False)
    def variable_rollout(previous, current, memory, commands, counts, mean, std, weights):
        previous = previous.copy()
        current = current.copy()
        memory = memory.copy()
        output = np.empty((len(commands), len(si)), np.float32)
        for k in range(len(commands)):
            current[0] = (commands[k, 0] - mean[0]) / std[0]
            current[16] = (commands[k, 1] - mean[16]) / std[16]
            for _ in range(counts[k]):
                state, memory = step(previous, current, memory, *weights)
                previous = current.copy()
                for j in range(len(si)):
                    current[si[j]] = state[j]
            for j in range(len(si)):
                output[k, j] = current[si[j]]
        return output, previous, current, memory

    @njit(cache=False)
    def dense_commit(previous, current, memory, command, count, mean, std, weights):
        previous = previous.copy()
        current = current.copy()
        memory = memory.copy()
        output = np.empty((count, len(si)), np.float64)
        current[0] = (command[0] - mean[0]) / std[0]
        current[16] = (command[1] - mean[16]) / std[16]
        for k in range(count):
            state, memory = step(previous, current, memory, *weights)
            previous = current.copy()
            for j in range(len(si)):
                current[si[j]] = state[j]
                output[k, j] = np.float64(state[j]) * std[si[j]] + mean[si[j]]
        return output, previous, current, memory
    return variable_rollout, dense_commit


class FullRecordingSession:
    def __init__(self, module, out, x, mean, std, dt, block, horizon, trainer, geometry):
        self.m = module
        self.out = Path(out)
        self.x = x
        self.mean = mean
        self.std = std
        self.dt = dt
        self.block = block
        self.horizon = horizon
        self.geometry = geometry
        self.count = len(x) - 2
        self.counts = np.full(int(np.ceil(self.count / block)), block, np.int64)
        self.counts[-1] = self.count - block * (len(self.counts) - 1)
        self.extended_counts = np.r_[self.counts, np.full(horizon, block, np.int64)]
        self.rows = 1 + np.r_[0, np.cumsum(self.extended_counts)]
        self.reference = np.asarray(x[np.minimum(self.rows, len(x) - 1)][:, module.MECH_INPUT], float) * std[module.MECH_INPUT] + mean[module.MECH_INPUT]
        self.reference[len(self.counts) + 1:, 2:] = 0.
        self.rollout_kernel, self.dense_kernel = kernels(trainer, module.SI)
        self.plan_counts = self.extended_counts[:horizon].copy()
        self.prediction = self.mapped('micro_predictor_states.npy', (self.count, 22))
        self.plant = self.mapped('micro_plant_states.npy', (self.count, 4))
        self.target = self.mapped('micro_reference_states.npy', (self.count, 4))
        self.actual_tip = self.mapped('micro_plant_tip.npy', (self.count, 4)) if geometry else None
        self.target_tip = self.mapped('micro_reference_tip.npy', (self.count, 4)) if geometry else None
        self.prior_tip = self.mapped('micro_predictor_tip.npy', (self.count, 4)) if geometry else None
        self.cursor = 0
        self.track_meter = StreamMetrics(module.MECHANICAL_COLS)
        self.mismatch_meter = StreamMetrics(module.MECHANICAL_COLS)
        self.tip_meter = StreamMetrics(['tip_x[m]', 'tip_z[m]', 'tip_vx[m/s]', 'tip_vz[m/s]']) if geometry else None
        self.tip_mismatch = StreamMetrics(self.tip_meter.names) if geometry else None
        self.lift_meter = StreamMetrics(['lift_tip_x[m]', 'lift_tip_z[m]', 'lift_tip_vx[m/s]', 'lift_tip_vz[m/s]']) if geometry else None
        self.distance_sse = 0.
        self.distance_sum = 0.
        self.distance_max = 0.
        self.lift_distance = np.zeros(3)
        self.model_distance = np.zeros(3)
        self.external_reference = False
        self.m.write_json(self.out / 'recording_coverage.json', dict(full_retained_recording=False, expected_microsteps=self.count, scored_microsteps=0, status='not_completed'))

    def mapped(self, name, shape):
        return np.lib.format.open_memmap(self.out / name, mode='w+', dtype=np.float64, shape=shape)

    def set_step(self, k):
        self.plan_counts = self.extended_counts[k:k + self.horizon]

    def rollout(self, previous, current, memory, commands, ignored_block, mean, std, weights):
        return self.rollout_kernel(previous, current, memory, commands, self.plan_counts[:len(commands)], mean, std, weights)

    def replay(self, plant, source, start, steps, block, mean, std, report_out=None, angle_tol=.002, velocity_tol=.002, low=None, high=None):
        initial = source[:2].astype(float) * std + mean
        mapping = [self.m.INPUT_COLS.index(col) for col in self.m.PLANT_INPUT_COLS]
        plant.reset(initial[:, mapping])
        snapshot = plant.snapshot()
        p, c, z = snapshot
        held_p, held_c, held_z = (value.copy() for value in snapshot)
        exact = np.empty((len(self.counts) + 1, 4))
        held = np.empty((len(self.extended_counts) + 1, 4))
        exact[0] = held[0] = plant.state()
        commands = np.empty((len(self.counts), 2))
        for k, size in enumerate(self.extended_counts):
            row = min(int(self.rows[k]), len(source) - 1)
            action = source[row, self.m.CI].astype(float) * std[self.m.CI] + mean[self.m.CI]
            rows = np.zeros((int(size), 6), np.float32)
            rows[:, 4:] = (action - plant.mean[4:]) / plant.std[4:]
            values, held_p, held_c, held_z = plant_chunk(rows, held_p, held_c, held_z, True, plant.mean, plant.std, *plant.weights)
            if not np.isfinite(values).all():
                raise RuntimeError(f'Full-recording reference plant diverged in control block {k}')
            held[k + 1] = values[-1]
            if k < len(self.counts):
                lo = int(self.rows[k]) - 1
                self.target[lo:lo + int(size)] = values
                commands[k] = action
                raw_commands = source[row:row + int(size)][:, self.m.CI].astype(float) * std[self.m.CI] + mean[self.m.CI]
                rows[:, 4:] = (raw_commands - plant.mean[4:]) / plant.std[4:]
                values, p, c, z = plant_chunk(rows, p, c, z, True, plant.mean, plant.std, *plant.weights)
                exact[k + 1] = values[-1]
            if k % 5000 == 0 or k + 1 == len(self.extended_counts):
                print(f'Full-CSV reference replay {k + 1:,}/{len(self.extended_counts):,} control blocks', flush=True)
        plant.restore(snapshot)
        self.held_reference = held
        self.target.flush()
        measured = self.reference[:len(self.counts) + 1]
        tables = []
        for label, values in [('exact_prepared_commands', exact), ('held_recorded_commands', held[:len(self.counts) + 1])]:
            table = self.m.tracking_metrics(measured[1:], values[1:])
            table.insert(0, 'replay', label)
            tables.append(table)
        metrics = pd.concat(tables, ignore_index=True)
        tolerance = [angle_tol, angle_tol, velocity_tol, velocity_tol]
        exact_ok = bool(np.all(tables[0].RMSE.to_numpy() <= tolerance))
        held_ok = bool(np.all(tables[1].RMSE.to_numpy() <= tolerance))
        in_bounds = low is None or bool(np.all(commands >= low - 1e-6) and np.all(commands <= high + 1e-6))
        summary = dict(exact_reproduction_pass=exact_ok, held_reproduction_pass=held_ok,
                       held_commands_in_mpc_bounds=in_bounds, reproduction_gate_pass=held_ok and in_bounds,
                       angle_rmse_tolerance_rad=angle_tol, velocity_rmse_tolerance_rad_s=velocity_tol,
                       full_retained_recording=True, microsteps=self.count, control_steps=len(self.counts),
                       last_control_block_microsteps=int(self.counts[-1]),
                       metrics=json.loads(metrics.to_json(orient='records')),
                       lookahead_after_csv_end='recorded target holds final pose and zero velocity; plant-generated target continues final held command')
        if report_out is not None:
            Path(report_out).mkdir(parents=True, exist_ok=True)
            metrics.to_csv(Path(report_out) / 'reproduction_metrics.csv', index=False)
            self.m.write_json(Path(report_out) / 'summary.json', summary)
        print(metrics.to_string(index=False), flush=True)
        return summary, measured, exact, held[:len(self.counts) + 1], commands, snapshot

    def choose_reference(self, mode):
        if mode == 'plant-generated':
            self.reference = self.held_reference
        else:
            for begin in range(0, self.count, 100000):
                end = min(begin + 100000, self.count)
                self.target[begin:end] = np.asarray(self.x[begin + 2:end + 2][:, self.m.MECH_INPUT], float) * self.std[self.m.MECH_INPUT] + self.mean[self.m.MECH_INPUT]

    def external_target(self, path):
        if self.geometry is None:
            raise ValueError('Cartesian reference requires geometry')
        values = read_cartesian_reference(path, np.arange(self.count + 1) * self.dt)
        self.target_tip[:] = values[1:]
        self.external_reference = True
        endpoint_indices = np.minimum(self.rows - 1, self.count)
        target = values[endpoint_indices].copy()
        target[self.rows - 1 > self.count, 2:] = 0.
        return target

    def advance(self, previous, current, memory, action, mean, std, weights, plant, k):
        size = int(self.counts[k])
        prior, pred_previous, pred_current, pred_memory = self.dense_kernel(previous, current, memory, action, size, mean, std, weights)
        rows = np.zeros((size, 6), np.float32)
        rows[:, 4:] = (action - plant.mean[4:]) / plant.std[4:]
        actual, plant.previous, plant.current, plant.latent = plant_chunk(rows, plant.previous, plant.current, plant.latent, True, plant.mean, plant.std, *plant.weights)
        if not np.isfinite(prior).all() or not np.isfinite(actual).all():
            raise RuntimeError(f'Committed full-CSV rollout diverged at microstep {self.cursor}')
        begin, end = self.cursor, self.cursor + size
        target = np.asarray(self.target[begin:end])
        self.prediction[begin:end] = prior
        self.plant[begin:end] = actual
        self.track_meter.add(target, actual)
        self.mismatch_meter.add(actual, prior[:, self.m.MECH_OUTPUT])
        if self.geometry:
            desired_lift, desired_tip = self.geometry.evaluate(target)
            actual_lift, actual_tip = self.geometry.evaluate(actual)
            _, prediction_tip = self.geometry.evaluate(prior[:, self.m.MECH_OUTPUT])
            desired = np.asarray(self.target_tip[begin:end]) if self.external_reference else desired_tip
            self.target_tip[begin:end] = desired
            self.actual_tip[begin:end] = actual_tip
            self.prior_tip[begin:end] = prediction_tip
            self.tip_meter.add(desired, actual_tip)
            self.tip_mismatch.add(actual_tip, prediction_tip)
            self.lift_meter.add(desired_lift, actual_lift)
            for accumulator, error in [(self.lift_distance, actual_lift[:, :2] - desired_lift[:, :2]), (self.model_distance, prediction_tip[:, :2] - actual_tip[:, :2])]:
                magnitudes = np.linalg.norm(error, axis=1)
                accumulator[0] += (magnitudes ** 2).sum()
                accumulator[1] += magnitudes.sum()
                accumulator[2] = max(accumulator[2], magnitudes.max())
            distance = np.linalg.norm(actual_tip[:, :2] - desired[:, :2], axis=1)
            self.distance_sse += (distance ** 2).sum()
            self.distance_sum += distance.sum()
            self.distance_max = max(self.distance_max, float(distance.max()))
            self.final_distance = float(distance[-1])
        self.cursor = end
        if k % 1000 == 0:
            self.m.write_json(self.out / 'recording_coverage.json', dict(full_retained_recording=False, expected_microsteps=self.count, scored_microsteps=self.cursor, status='in_progress'))
        observed_previous = plant.previous[:4].astype(float) * plant.std[:4] + plant.mean[:4]
        observed_current = plant.current[:4].astype(float) * plant.std[:4] + plant.mean[:4]
        prior_full = pred_current.astype(float) * std + mean
        pred_previous[self.m.MECH_INPUT] = (observed_previous - mean[self.m.MECH_INPUT]) / std[self.m.MECH_INPUT]
        pred_current[self.m.MECH_INPUT] = (observed_current - mean[self.m.MECH_INPUT]) / std[self.m.MECH_INPUT]
        return pred_previous, pred_current, pred_memory, prior[-1, self.m.MECH_OUTPUT], actual[-1], prior_full

    def finish(self, offset):
        if self.cursor != self.count:
            raise RuntimeError('Full recording was not completed; do not treat partial arrays as full-CSV results')
        for array in (self.prediction, self.plant, self.target, self.actual_tip, self.target_tip, self.prior_tip):
            if array is not None:
                array.flush()
        coverage = dict(full_retained_recording=True, retained_rows=len(self.x),
                        initialization_rows=2, predicted_and_scored_microsteps=self.cursor,
                        expected_microsteps=self.count, control_updates=len(self.counts),
                        internal_sample_time_s=self.dt, control_block_samples=self.block,
                        last_control_block_samples=int(self.counts[-1]),
                        first_prediction_csv_time_s=offset + 2 * self.dt,
                        last_prediction_csv_time_s=offset + (len(self.x) - 1) * self.dt,
                        startup_rows_excluded=int(round(offset / self.dt)),
                        metrics_resolution='every internal microstep, not just control endpoints')
        self.m.write_json(self.out / 'recording_coverage.json', coverage)
        self.track_meter.table().to_csv(self.out / 'tracking_metrics.csv', index=False)
        self.mismatch_meter.table().to_csv(self.out / 'model_mismatch_metrics.csv', index=False)
        distance = None
        if self.geometry:
            self.tip_meter.table().to_csv(self.out / 'end_effector_metrics.csv', index=False)
            self.tip_mismatch.table().to_csv(self.out / 'end_effector_mismatch_metrics.csv', index=False)
            pd.concat([self.lift_meter.table().assign(boom='lift'), self.tip_meter.table().assign(boom='jib/end_effector')], ignore_index=True).to_csv(self.out / 'boom_tip_metrics.csv', index=False)
            distance = dict(position_RMSE_m=float(np.sqrt(self.distance_sse / self.count)),
                            position_MAE_m=float(self.distance_sum / self.count),
                            position_max_error_m=self.distance_max, final_position_error_m=self.final_distance,
                            scored_microsteps=self.count)
            self.m.write_json(self.out / 'full_recording_kinematics_summary.json', distance)
            print('All-microstep end-effector metrics:\n' + self.tip_meter.table().to_string(index=False), flush=True)
            print('All-microstep distance metrics: ' + json.dumps(distance), flush=True)
        return coverage, distance

    def plots(self, offset):
        import matplotlib.pyplot as plt
        first = offset + 2 * self.dt
        for selected, title, filename in [([0, 1, 2, 3], 'Both booms: whole CSV', 'both_booms_full_csv.png'), ([0, 2], 'Lift boom: whole CSV', 'lift_full_csv.png'), ([1, 3], 'Jib boom: whole CSV', 'jib_full_csv.png')]:
            fig, axes = plt.subplots(len(selected), 1, figsize=(12, 3 * len(selected)), layout='constrained')
            for ax, j in zip(np.atleast_1d(axes), selected):
                plot_recording_pair(ax, scalar_source(self.target, j), scalar_source(self.plant, j), self.count, self.dt, first, 'Mechanical plant')
                ax.set_title(self.m.MECHANICAL_COLS[j])
                ax.legend(fontsize=8)
                ax.grid(alpha=.3)
            fig.suptitle(f'{title}; all {self.count:,} internal samples included')
            fig.savefig(self.out / filename, dpi=130)
        if self.geometry:
            fig, axes = plt.subplots(4, 1, figsize=(12, 12), layout='constrained')
            for j, ax in enumerate(axes):
                plot_recording_pair(ax, scalar_source(self.target_tip, j), scalar_source(self.actual_tip, j), self.count, self.dt, first, 'Plant end effector')
                ax.set_title(self.tip_meter.names[j])
                ax.legend(fontsize=8)
                ax.grid(alpha=.3)
            fig.suptitle(f'End-effector tracking across the entire retained CSV: {self.count:,} samples')
            fig.savefig(self.out / 'end_effector_full_csv.png', dpi=130)


def show_full_results(module, folder, show, geometry):
    folder = Path(folder)
    coverage = json.loads((folder / 'recording_coverage.json').read_text(encoding='utf-8'))
    if not coverage.get('full_retained_recording'):
        raise ValueError('This full-recording run is incomplete; inspect its progress and diagnostics')
    print((folder / 'results.txt').read_text(encoding='utf-8'), flush=True)
    import matplotlib
    matplotlib.use('TkAgg' if show else 'Agg')
    import matplotlib.pyplot as plt
    session = FullRecordingSession.__new__(FullRecordingSession)
    session.m = module
    session.out = folder
    session.count = coverage['expected_microsteps']
    session.dt = coverage['internal_sample_time_s']
    session.geometry = geometry
    session.target = np.load(folder / 'micro_reference_states.npy', mmap_mode='r')
    session.plant = np.load(folder / 'micro_plant_states.npy', mmap_mode='r')
    if geometry:
        session.actual_tip = np.load(folder / 'micro_plant_tip.npy', mmap_mode='r')
        session.target_tip = np.load(folder / 'micro_reference_tip.npy', mmap_mode='r')
        session.tip_meter = StreamMetrics(['tip_x[m]', 'tip_z[m]', 'tip_vx[m/s]', 'tip_vz[m/s]'])
    session.plots(coverage['startup_rows_excluded'] * session.dt)
    with np.load(folder / 'mpc_mechanical_plant_arrays.npz') as arrays:
        config = module.read_report(folder / 'config.json')
        module.make_plots(folder, arrays['time_seconds'], arrays['reference_physical'], arrays['plant_mechanical_physical'], arrays['predictor_one_interval_mechanical_physical'], arrays['applied_commands_physical'], arrays['solve_seconds'], arrays['cost'], config['control_interval_seconds'], np.asarray(config['command_bounds']['low']), np.asarray(config['command_bounds']['high']), show, arrays['predictor_horizons_mechanical_physical'], arrays['teacher_reference_horizons_physical'], arrays['recorded_commands_physical'], geometry, arrays['reference_end_effector_physical'] if 'reference_end_effector_physical' in arrays.files else None)
    plt.close('all')
