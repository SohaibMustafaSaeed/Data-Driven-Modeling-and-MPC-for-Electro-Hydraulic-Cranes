from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlanarGeometry:
    lift_length: float
    jib_length: float
    jib_angle_mode: str
    base_x: float = 0.0
    base_z: float = 0.0
    lift_offset_deg: float = 0.0
    jib_offset_deg: float = 0.0
    lift_sign: float = 1.0
    jib_sign: float = 1.0
    lift_vector: tuple | None = None
    jib_vector: tuple | None = None

    def __post_init__(self):
        numbers = [value for key, value in asdict(self).items() if key not in ('jib_angle_mode', 'lift_vector', 'jib_vector')]
        if not np.isfinite(numbers).all() or min(self.lift_length, self.jib_length) <= 0:
            raise ValueError('Boom lengths must be positive metres; geometry values must be finite')
        if self.jib_angle_mode not in ('relative', 'absolute'):
            raise ValueError('jib_angle_mode must be relative or absolute')
        if self.lift_sign not in (-1, 1) or self.jib_sign not in (-1, 1):
            raise ValueError('Angle signs must be +1 or -1')
        for name in ('lift', 'jib'):
            vector = getattr(self, name + '_vector')
            if vector is not None:
                values = np.asarray(vector, float)
                if values.shape != (2,) or not np.isfinite(values).all() or np.linalg.norm(values) <= 0:
                    raise ValueError('Local link vectors require two finite coordinates and nonzero length')
                object.__setattr__(self, name + '_vector', tuple(values.tolist()))
                object.__setattr__(self, name + '_length', float(np.linalg.norm(values)))

    def evaluate(self, mechanics):
        state = np.asarray(mechanics, dtype=float)
        if state.shape[-1] != 4:
            raise ValueError('Expected lift angle, jib angle, lift velocity, jib velocity')
        lift = self.lift_sign * state[..., 0] + np.deg2rad(self.lift_offset_deg)
        jib = self.jib_sign * state[..., 1] + np.deg2rad(self.jib_offset_deg)
        lift_rate = self.lift_sign * state[..., 2]
        jib_rate = self.jib_sign * state[..., 3]
        if self.jib_angle_mode == 'relative':
            jib = lift + jib
            jib_rate = lift_rate + jib_rate
        def rotate(angle, local):
            return np.stack((local[0] * np.cos(angle) - local[1] * np.sin(angle), local[0] * np.sin(angle) + local[1] * np.cos(angle)), axis=-1)
        lift_vector = rotate(lift, self.lift_vector or (self.lift_length, 0.0))
        jib_vector = rotate(jib, self.jib_vector or (self.jib_length, 0.0))
        lift_velocity = lift_rate[..., None] * np.stack((-lift_vector[..., 1], lift_vector[..., 0]), axis=-1)
        jib_velocity = jib_rate[..., None] * np.stack((-jib_vector[..., 1], jib_vector[..., 0]), axis=-1)
        elbow = lift_vector + np.asarray((self.base_x, self.base_z))
        tip = elbow + jib_vector
        return np.concatenate((elbow, lift_velocity), axis=-1), np.concatenate((tip, lift_velocity + jib_velocity), axis=-1)

    def end_effector(self, mechanics):
        return self.evaluate(mechanics)[1]


def add_geometry_arguments(parser):
    parser.add_argument('--tracking-space', choices=['joint', 'cartesian'], default='cartesian')
    parser.add_argument('--geometry-file', type=Path)
    parser.add_argument('--lift-length', type=float, help='Lift joint to elbow distance in metres')
    parser.add_argument('--jib-length', type=float, help='Elbow to end-effector distance in metres')
    parser.add_argument('--jib-angle-mode', choices=['relative', 'absolute'], help='Jib angle relative to lift boom, or absolute to horizontal')
    parser.add_argument('--lift-local-vector', type=float, nargs=2, metavar=('X', 'Z'))
    parser.add_argument('--jib-local-vector', type=float, nargs=2, metavar=('X', 'Z'))
    parser.add_argument('--base-position', type=float, nargs=2, metavar=('X', 'Z'))
    parser.add_argument('--angle-offsets-deg', type=float, nargs=2, metavar=('LIFT', 'JIB'))
    parser.add_argument('--angle-signs', type=float, nargs=2, metavar=('LIFT', 'JIB'))
    parser.add_argument('--position-scale', type=float, default=0.01, help='Cartesian position error normalization in metres')
    parser.add_argument('--cartesian-velocity-scale', type=float, default=0.1, help='Cartesian velocity normalization in m/s')
    parser.add_argument('--position-weight', type=float, default=1.0)
    parser.add_argument('--cartesian-velocity-weight', type=float, default=0.1)
    parser.add_argument('--terminal-position-weight', type=float, default=5.0)
    parser.add_argument('--terminal-cartesian-velocity-weight', type=float, default=1.0)
    parser.add_argument('--cartesian-reference', type=Path, help='Optional reachable trajectory CSV: time_s,x_m,z_m; vx_m_s,vz_m_s optional')


def geometry_from_args(args, parser, saved=None):
    values = dict(saved or {})
    default_geometry = Path(__file__).with_name('geometry_patu.json')
    if not values and default_geometry.is_file():
        values.update(json.loads(default_geometry.read_text(encoding='utf-8')))
    if args.geometry_file:
        values.update(json.loads(args.geometry_file.read_text(encoding='utf-8')))
    for key in ('lift_length', 'jib_length', 'jib_angle_mode'):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
            if key in ('lift_length', 'jib_length'):
                values.pop(key.replace('_length', '_vector'), None)
    for name in ('lift', 'jib'):
        value = getattr(args, name + '_local_vector')
        if value is not None:
            values[name + '_vector'] = value
        if values.get(name + '_vector') is not None:
            values[name + '_length'] = float(np.linalg.norm(values[name + '_vector']))
    for option, keys in [('base_position', ('base_x', 'base_z')), ('angle_offsets_deg', ('lift_offset_deg', 'jib_offset_deg')), ('angle_signs', ('lift_sign', 'jib_sign'))]:
        value = getattr(args, option)
        if value is not None:
            values.update(zip(keys, value))
    if not values:
        if args.tracking_space == 'cartesian' or args.cartesian_reference:
            parser.error('Cartesian tracking requires --lift-length, --jib-length and --jib-angle-mode, or a completed --geometry-file')
        return None
    missing = [key for key in ('lift_length', 'jib_length', 'jib_angle_mode') if values.get(key) is None]
    if missing:
        parser.error('Missing crane geometry: ' + ', '.join(missing))
    try:
        return PlanarGeometry(**values)
    except (TypeError, ValueError) as error:
        parser.error(str(error))


def read_cartesian_reference(path, times):
    table = pd.read_csv(path)
    required = ('time_s', 'x_m', 'z_m')
    if not all(key in table for key in required):
        raise ValueError('Cartesian reference CSV requires time_s,x_m,z_m')
    data = table[list(required)].to_numpy(dtype=float)
    if len(data) < 2 or not np.isfinite(data).all() or not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError('Reference needs finite positions and strictly increasing times')
    if times[0] < data[0, 0] - 1e-12 or times[-1] > data[-1, 0] + 1e-12:
        raise ValueError('Cartesian reference must cover the entire episode plus prediction lookahead')
    position = np.column_stack([np.interp(times, data[:, 0], data[:, j]) for j in (1, 2)])
    velocity_columns = ('vx_m_s', 'vz_m_s')
    if any(key in table for key in velocity_columns) and not all(key in table for key in velocity_columns):
        raise ValueError('Supply both vx_m_s and vz_m_s or neither')
    if all(key in table for key in velocity_columns):
        velocity = table[list(velocity_columns)].to_numpy(dtype=float)
        if not np.isfinite(velocity).all():
            raise ValueError('Reference velocities must be finite')
        rates = np.column_stack([np.interp(times, data[:, 0], velocity[:, j]) for j in range(2)])
    else:
        rates = np.gradient(position, times, axis=0)
    return np.column_stack((position, rates))


def axis_metrics(reference, prediction, names):
    reference = np.asarray(reference, float)
    prediction = np.asarray(prediction, float)
    error = prediction - reference
    mse = np.mean(error ** 2, axis=0)
    variance = np.var(reference, axis=0)
    threshold = np.finfo(float).eps * np.maximum(1.0, np.mean(reference, axis=0) ** 2 + variance)
    ratio = np.full(error.shape[1], np.nan)
    np.divide(mse, variance, out=ratio, where=variance > threshold)
    return pd.DataFrame(dict(feature=names, RMSE=np.sqrt(mse), MAE=np.mean(abs(error), axis=0), max_abs_error=np.max(abs(error), axis=0), R2=1 - ratio))


def distance_metrics(reference, prediction):
    distance = np.linalg.norm(prediction[:, :2] - reference[:, :2], axis=1)
    return dict(position_RMSE_m=float(np.sqrt(np.mean(distance ** 2))), position_MAE_m=float(distance.mean()), position_max_error_m=float(distance.max()), final_position_error_m=float(distance[-1]))


def save_kinematic_results(out, times, reference, plant, prediction, geometry, desired_tip=None, tracking_space='joint'):
    out = Path(out)
    ref_elbow, reference_tip = geometry.evaluate(reference)
    actual_elbow, actual_tip = geometry.evaluate(plant)
    pred_elbow, prediction_tip = geometry.evaluate(prediction)
    if desired_tip is not None:
        reference_tip = np.asarray(desired_tip, float)
    names = ['tip_x[m]', 'tip_z[m]', 'tip_vx[m/s]', 'tip_vz[m/s]']
    tip_metrics = axis_metrics(reference_tip[1:], actual_tip[1:], names)
    tip_metrics.to_csv(out / 'end_effector_metrics.csv', index=False)
    mismatch = axis_metrics(actual_tip[1:], prediction_tip[1:], names)
    mismatch.to_csv(out / 'end_effector_mismatch_metrics.csv', index=False)
    elbow_metrics = axis_metrics(ref_elbow[1:], actual_elbow[1:], [name.replace('tip_', 'lift_tip_') for name in names])
    combined = pd.concat([elbow_metrics.assign(boom='lift'), tip_metrics.assign(boom='jib/end_effector')], ignore_index=True)
    combined.to_csv(out / 'boom_tip_metrics.csv', index=False)
    summary = dict(tracking_space=tracking_space, geometry=asdict(geometry), end_effector=distance_metrics(reference_tip[1:], actual_tip[1:]), lift_tip=distance_metrics(ref_elbow[1:], actual_elbow[1:]), one_interval_model_mismatch=distance_metrics(actual_tip[1:], prediction_tip[1:]), initial_measured_sample_excluded=True, planar_axes='horizontal x, vertical z; metres; no slew or flexible deflection')
    external = not np.allclose(reference_tip, geometry.end_effector(reference), rtol=0.0, atol=1e-10)
    summary['lift_tip_reference_role'] = 'mechanical guide, not inverse kinematics of external Cartesian target' if external else 'reference elbow from mechanical reference'
    (out / 'kinematics_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'kinematics_config.json').write_text(json.dumps(asdict(geometry), indent=2), encoding='utf-8')
    trace = {'time_s': times}
    for prefix, values in [('reference_tip', reference_tip), ('plant_tip', actual_tip), ('predictor_tip', prediction_tip), ('reference_lift_tip', ref_elbow), ('plant_lift_tip', actual_elbow)]:
        for j, name in enumerate(('x_m', 'z_m', 'vx_m_s', 'vz_m_s')):
            trace[f'{prefix}_{name}'] = values[:, j]
    pd.DataFrame(trace).to_csv(out / 'trajectory_cartesian.csv', index=False)
    np.savez(out / 'kinematic_trajectories.npz', time_seconds=times, reference_end_effector=reference_tip, plant_end_effector=actual_tip, predictor_end_effector=prediction_tip, reference_lift_tip=ref_elbow, plant_lift_tip=actual_elbow, predictor_lift_tip=pred_elbow)
    text = '\nEnd-effector tracking (metres and m/s):\n' + tip_metrics.to_string(index=False, float_format=lambda value: f'{value:.8g}')
    text += '\nBoom-tip metrics:\n' + combined.to_string(index=False, float_format=lambda value: f'{value:.8g}')
    text += '\nCartesian distance metrics:\n' + json.dumps(summary, indent=2)
    print(text, flush=True)
    (out / 'kinematics_results.txt').write_text(text, encoding='utf-8')
    return summary


def make_boom_plots(out, times, reference, plant, prediction, commands, geometry=None, desired_tip=None):
    import matplotlib.pyplot as plt
    external = geometry is not None and desired_tip is not None and not np.allclose(desired_tip, geometry.end_effector(reference), rtol=0.0, atol=1e-10)
    joint_label = 'Joint guide (external Cartesian target)' if external else 'Joint reference'
    for boom, selected, command in [('lift', (0, 2), 0), ('jib', (1, 3), 1)]:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout='constrained')
        fig.canvas.manager.set_window_title(f'{boom.title()} boom: angle, velocity and command')
        for ax, j, unit in zip(axes[:2], selected, ('rad', 'rad/s')):
            ax.plot(times, reference[:, j], color='black', label=joint_label)
            ax.plot(times, prediction[:, j], '--', color='tab:orange', label='One-interval prediction')
            ax.plot(times, plant[:, j], color='tab:blue', label='Mechanical plant')
            ax.set_ylabel(unit)
            ax.legend()
            ax.grid(alpha=.3)
        axes[2].step(times[:-1], commands[:, command], where='post', label=f'{boom.title()} command')
        axes[2].set_ylabel('V')
        axes[2].set_xlabel('Episode time [s]')
        axes[2].grid(alpha=.3)
        fig.suptitle(f'{boom.title()} boom')
        fig.savefig(Path(out) / f'{boom}_boom_states.png', dpi=130)
    if geometry is None:
        return
    ref_elbow, ref_tip = geometry.evaluate(reference)
    plant_elbow, plant_tip = geometry.evaluate(plant)
    pred_elbow, pred_tip = geometry.evaluate(prediction)
    if desired_tip is not None:
        ref_tip = desired_tip
    paths = [('Lift boom tip', ref_elbow, plant_elbow, pred_elbow, 'lift'), ('Jib tip / end effector', ref_tip, plant_tip, pred_tip, 'jib')]
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    fig.canvas.manager.set_window_title('Both boom trajectories')
    for label, target, actual, prior, boom in paths:
        color = 'tab:blue' if boom == 'lift' else 'tab:green'
        ax.plot(target[:, 0], target[:, 1], '--', color=color, label=f'{label}: ' + ('joint guide' if external and boom == 'lift' else 'reference'))
        ax.plot(actual[:, 0], actual[:, 1], color=color, label=f'{label}: plant')
    base = np.array([geometry.base_x, geometry.base_z])
    final = np.vstack((base, plant_elbow[-1, :2], plant_tip[-1, :2]))
    ax.plot(final[:, 0], final[:, 1], 'o-', color='black', alpha=.6, label='Final two-boom configuration')
    ax.set(xlabel='Horizontal x [m]', ylabel='Vertical z [m]', title='Lift and jib trajectories together')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(Path(out) / 'both_boom_trajectories.png', dpi=130)
    for label, target, actual, prior, boom in paths:
        fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
        fig.canvas.manager.set_window_title(f'{label}: Cartesian trajectory')
        ax.plot(target[:, 0], target[:, 1], color='black', label='Joint guide' if external and boom == 'lift' else 'Reference')
        ax.plot(prior[:, 0], prior[:, 1], '--', color='tab:orange', label='One-interval prediction')
        ax.plot(actual[:, 0], actual[:, 1], color='tab:blue', label='Mechanical plant')
        ax.scatter(actual[0, 0], actual[0, 1], marker='o', color='tab:blue', label='Start')
        ax.scatter(actual[-1, 0], actual[-1, 1], marker='x', color='tab:blue', label='End')
        ax.set(xlabel='Horizontal x [m]', ylabel='Vertical z [m]', title=label)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.3)
        ax.legend()
        fig.savefig(Path(out) / f'{boom}_boom_trajectory.png', dpi=130)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, layout='constrained')
    fig.canvas.manager.set_window_title('End-effector tracking versus time')
    for j, ax in enumerate(axes.flat):
        ax.plot(times, ref_tip[:, j], color='black', label='Reference')
        ax.plot(times, pred_tip[:, j], '--', color='tab:orange', label='One-interval prediction')
        ax.plot(times, plant_tip[:, j], color='tab:blue', label='Mechanical plant')
        ax.set_title(('x [m]', 'z [m]', 'vx [m/s]', 'vz [m/s]')[j])
        ax.set_xlabel('Episode time [s]')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.savefig(Path(out) / 'end_effector_tracking.png', dpi=130)
