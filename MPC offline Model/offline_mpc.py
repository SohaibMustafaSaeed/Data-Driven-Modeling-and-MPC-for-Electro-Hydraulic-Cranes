import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, resolve_array, same_recording
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from numba import njit
from scipy.optimize import minimize
from common.data import DEFAULT_DATASET_DIR, INPUT_COLS, OUTPUT_COLS, COMMAND_INDICES, STATE_INDICES, write_json
from common.evaluation import load_test
from common.project import read_report
from dataclasses import asdict
from crane_kinematics import (PlanarGeometry, add_geometry_arguments, geometry_from_args,
                              read_cartesian_reference, save_kinematic_results, make_boom_plots)
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / 'results'
MECHANICAL_COLS = ('thetalift[rad]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]')
MECH_INPUT = np.asarray([INPUT_COLS.index(col) for col in MECHANICAL_COLS], np.int64)
MECH_OUTPUT = np.asarray([OUTPUT_COLS.index(col) for col in MECHANICAL_COLS], np.int64)
SI = np.asarray(STATE_INDICES, np.int64)
CI = np.asarray(COMMAND_INDICES, np.int64)
MECHANICAL_DIR = SCRIPT_DIR / 'plant'
DEFAULT_PLANT_CHECKPOINT = SCRIPT_DIR / 'artifacts/mechanical_plant/best_model.pt'
if str(MECHANICAL_DIR) not in sys.path:
    sys.path.insert(0, str(MECHANICAL_DIR))
from mechanical_compact import MechanicalPlant, held_command as warm_plant_kernel, INPUT_COLS as PLANT_INPUT_COLS

def default_checkpoint():
    checkpoint = ROOT / 'GRU Latent/artifacts/best_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError('GRU checkpoint missing; train GRU Latent/gru2.py first')
    return checkpoint

def load_surrogate(checkpoint):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    modules = {'gru2_encoded_residual_numba_24x22_v2': 'gru2', 'lstm4_encoded_residual_numba_24x22_v1': 'lstm4', 'causal_window_residual_transformer_numba_24x22_v1': 'transformer', 'compact_latent_recurrent_numba_24x22_v1': 'trainCompactLatent'}
    module = modules.get(ck.get('format'))
    if module is None:
        raise ValueError(f"Unsupported full-state checkpoint format: {ck.get('format')}")
    trainer = importlib.import_module(module)
    model, ck = trainer.load_bundle(checkpoint)
    return (trainer, model, ck)

def make_blocked_rollout(trainer):
    step = trainer.native_step

    @njit(cache=False)
    def rollout(previous, current, memory, commands, block_steps, mean, std, weights):
        previous = previous.copy()
        current = current.copy()
        memory = memory.copy()
        states = np.empty((len(commands), 22), np.float32)
        for k in range(len(commands)):
            current[0] = (commands[k, 0] - mean[0]) / std[0]
            current[16] = (commands[k, 1] - mean[16]) / std[16]
            for _ in range(block_steps):
                state, memory = step(previous, current, memory, *weights)
                previous = current.copy()
                for j in range(22):
                    current[SI[j]] = state[j]
            for j in range(22):
                states[k, j] = current[SI[j]]
        return (states, previous, current, memory)
    return rollout

def sampled_training_bounds(ck, mean, std, samples_per_file=1000, dataset_dir=DEFAULT_DATASET_DIR):
    low = np.full(2, np.inf)
    high = np.full(2, -np.inf)
    for entry in ck['training_config']['training_manifest']:
        path = resolve_array(entry, dataset_dir)
        if not path.is_absolute():
            path = dataset_dir / path
        array = np.load(path, mmap_mode='r')
        count = int(entry['train_rows'])
        ids = np.unique(np.linspace(0, count - 1, min(count, samples_per_file), dtype=np.int64))
        commands = np.asarray(array[ids][:, CI], dtype=float) * std[CI] + mean[CI]
        low = np.minimum(low, commands.min(0))
        high = np.maximum(high, commands.max(0))
    return (low, high)

def select_segment(x, episode_steps, block, horizon, dt, start_seconds):
    margin = (episode_steps + horizon) * block
    last_start = len(x) - 1 - margin
    if last_start < 1:
        raise ValueError('Recording is too short for episode plus MPC lookahead')
    if start_seconds is not None:
        start = max(1, int(round(start_seconds / dt)))
        if not 1 <= start <= last_start:
            raise ValueError(f'start-seconds is relative to retained data and must allow a previous row and full lookahead; maximum {last_start * dt:g}s')
        return start
    stride = max(block, int(round(0.01 / dt)))
    indices = np.arange(1, len(x) - horizon * block, stride, dtype=np.int64)
    return (indices, last_start, stride)

def activity_segment(x, selection, mean, std, episode_steps, block):
    if isinstance(selection, (int, np.integer)):
        return int(selection)
    indices, last_start, stride = selection
    velocity = np.asarray(x[indices][:, MECH_INPUT[2:]], float) * std[MECH_INPUT[2:]] + mean[MECH_INPUT[2:]]
    window = max(1, int(math.ceil(episode_steps * block / stride)))
    if len(indices) < window:
        raise ValueError('Not enough rows for reference segment')
    rms = []
    for j in range(2):
        prefix = np.r_[0.0, np.cumsum(velocity[:, j] ** 2)]
        rms.append(np.sqrt(np.maximum((prefix[window:] - prefix[:-window]) / window, 0)))
    score = np.minimum(rms[0], rms[1]) + 0.1 * (rms[0] + rms[1])
    starts = indices[:len(score)]
    valid = starts <= last_start
    if not valid.any():
        raise ValueError('No complete automatic reference segment')
    return int(starts[valid][np.argmax(score[valid])])

def expand_moves(decision, assignment, center, halfwidth):
    moves = np.asarray(decision, float).reshape(-1, 2)
    return center + halfwidth * moves[assignment]

class MPCObjective:

    def __init__(self, rollout, previous, current, memory, reference, previous_command, center, halfwidth, neutral, assignment, block, mean, std, weights, cost_weights, terminal_angle_weight=0.0, terminal_velocity_weight=0.0, tracking_scales=None, geometry=None, cartesian_reference=None, cartesian_weights=None, cartesian_scales=None):
        self.rollout = rollout
        self.previous = previous
        self.current = current
        self.memory = memory
        self.reference = np.asarray(reference, float)
        self.previous_command = previous_command
        self.center = center
        self.halfwidth = halfwidth
        self.neutral = neutral
        self.assignment = assignment
        self.block = block
        self.mean = mean
        self.std = std
        self.weights = weights
        self.cost_weights = np.r_[cost_weights, terminal_angle_weight, terminal_velocity_weight]
        self.error_ratio = np.ones(4) if tracking_scales is None else std[MECH_INPUT] / np.asarray(tracking_scales, float)
        self.geometry = geometry
        if geometry is not None:
            physical_reference = self.reference * std[MECH_INPUT] + mean[MECH_INPUT]
            self.cartesian_reference = geometry.end_effector(physical_reference) if cartesian_reference is None else np.asarray(cartesian_reference, float)
            self.cartesian_scales = np.ones(4) if cartesian_scales is None else np.asarray(cartesian_scales, float)
            selected_weights = np.asarray([1.0, 0.1, 5.0, 1.0] if cartesian_weights is None else cartesian_weights, float)
            if self.cartesian_reference.shape != self.reference.shape or self.cartesian_scales.shape != (4,) or not np.isfinite(self.cartesian_scales).all() or np.any(self.cartesian_scales <= 0):
                raise ValueError('Cartesian reference must match horizon x 4; scales must be positive and finite')
            if selected_weights.shape != (4,) or not np.isfinite(selected_weights).all() or np.any(selected_weights < 0):
                raise ValueError('Four Cartesian weights must be finite and nonnegative')
            self.cost_weights = np.r_[self.cost_weights, selected_weights]
        self.best_cost = np.inf
        self.best_decision = None
        self.best_components = None
        self.calls = 0
        self.invalid_calls = 0

    def __call__(self, decision):
        self.calls += 1
        commands = expand_moves(decision, self.assignment, self.center, self.halfwidth)
        states, _, cur, memory = self.rollout(self.previous, self.current, self.memory, commands, self.block, self.mean, self.std, self.weights)
        if not np.isfinite(states).all() or not np.isfinite(cur).all() or (not np.isfinite(memory).all()):
            self.invalid_calls += 1
            return 1000000000000.0
        error = (states[:, MECH_OUTPUT].astype(float) - self.reference) * self.error_ratio
        effort = (commands - self.neutral) / self.halfwidth
        change = np.diff(np.vstack((self.previous_command, commands)), axis=0) / self.halfwidth
        components = np.asarray([np.mean(error[:, :2] ** 2), np.mean(error[:, 2:] ** 2), np.mean(effort ** 2), np.mean(change ** 2), np.sum(error[-1, :2] ** 2), np.sum(error[-1, 2:] ** 2)])
        if self.geometry is not None:
            physical = states[:, MECH_OUTPUT].astype(float) * self.std[MECH_INPUT] + self.mean[MECH_INPUT]
            cartesian = (self.geometry.end_effector(physical) - self.cartesian_reference) / self.cartesian_scales
            components = np.r_[components, np.mean(cartesian[:, :2] ** 2), np.mean(cartesian[:, 2:] ** 2), np.sum(cartesian[-1, :2] ** 2), np.sum(cartesian[-1, 2:] ** 2)]
        cost = float(components @ self.cost_weights)
        if not math.isfinite(cost):
            self.invalid_calls += 1
            return 1000000000000.0
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_decision = np.asarray(decision, float).copy()
            self.best_components = components
        return cost

def tracking_metrics(reference, prediction):
    error = prediction - reference
    variance = np.var(reference, axis=0)
    threshold = np.finfo(float).eps * np.maximum(1.0, np.mean(reference, axis=0) ** 2 + variance)
    r2 = np.full(4, np.nan)
    usable = variance > threshold
    r2[usable] = 1.0 - np.mean(error[:, usable] ** 2, axis=0) / variance[usable]
    return pd.DataFrame(dict(feature=MECHANICAL_COLS, RMSE=np.sqrt(np.mean(error ** 2, axis=0)), MAE=np.mean(abs(error), axis=0), max_abs_error=abs(error).max(0), R2=r2))

def advance_and_correct(rollout, previous, current, memory, action, block, mean, std, weights, plant):
    _, pred_previous, pred_current, pred_memory = rollout(previous, current, memory, np.asarray(action, float)[None], block, mean, std, weights)
    if not np.isfinite(pred_current).all() or not np.isfinite(pred_memory).all():
        raise RuntimeError('Full predictor diverged on the committed action')
    prior_physical = pred_current.astype(float) * std + mean
    predicted_mechanics = prior_physical[MECH_INPUT].copy()
    plant_mechanics = plant.advance(action, internal_steps=block)
    plant_previous, plant_current, _ = plant.snapshot()
    observed_previous = plant_previous[:4].astype(float) * plant.std[:4] + plant.mean[:4]
    observed_current = plant_current[:4].astype(float) * plant.std[:4] + plant.mean[:4]
    corrected_previous = pred_previous.copy()
    corrected_current = pred_current.copy()
    corrected_previous[MECH_INPUT] = (observed_previous - mean[MECH_INPUT]) / std[MECH_INPUT]
    corrected_current[MECH_INPUT] = (observed_current - mean[MECH_INPUT]) / std[MECH_INPUT]
    return (corrected_previous, corrected_current, pred_memory, predicted_mechanics, plant_mechanics, prior_physical)

def require_completed(checkpoint, ck, parser, label):
    path = checkpoint.parent / 'training_history.json'
    expected = ck['training_config']['arguments'].get('epochs')
    if expected is not None and (not path.is_file() or len(json.loads(path.read_text())) < expected):
        parser.error(f'{label} training is incomplete: {checkpoint.parent}')

def make_plots(out, time_grid, reference, plant_states, predicted, commands, solve_times, costs, dt_control, low, high, show, horizon_forecasts=None, horizon_references=None, recorded_commands=None, geometry=None, desired_tip=None):
    import matplotlib
    matplotlib.use('TkAgg' if show else 'Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    external_reference = geometry is not None and desired_tip is not None and not np.allclose(desired_tip, geometry.end_effector(reference), rtol=0.0, atol=1e-10)
    joint_label = 'Joint guide (external Cartesian target)' if external_reference else 'Desired reference'
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout='constrained')
    for j, ax in enumerate(axes.flat):
        ax.plot(time_grid, reference[:, j], color='black', label=joint_label)
        ax.plot(time_grid, predicted[:, j], '--', color='tab:orange', label='Full surrogate: one-control-step forecast')
        ax.plot(time_grid, plant_states[:, j], color='tab:blue', label='Mechanical Compact plant')
        ax.set_title(MECHANICAL_COLS[j])
        ax.set_ylabel('rad' if j < 2 else 'rad/s')
        ax.set_xlabel('Episode time [s]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Full-state MPC controlling a separate mechanical surrogate plant')
    fig.savefig(out / 'reference_prediction_plant.png', dpi=150)
    if horizon_forecasts is not None:
        K, N, _ = horizon_forecasts.shape
        sample = min(K // 2, max(0, K - N))
        horizon_time = time_grid[sample] + np.arange(1, N + 1) * dt_control
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout='constrained')
        for j, ax in enumerate(axes.flat):
            ax.plot(horizon_time, horizon_references[sample, :, j], color='black', label='Joint guide horizon' if external_reference else 'Desired horizon')
            ax.plot(horizon_time, horizon_forecasts[sample, :, j], '--', color='tab:orange', label='Full-model fixed plan')
            available = min(N, K - sample)
            ax.plot(horizon_time[:available], plant_states[sample + 1:sample + 1 + available, j], color='tab:blue', label='Plant with later replanning')
            ax.set_title(MECHANICAL_COLS[j])
            ax.set_ylabel('rad' if j < 2 else 'rad/s')
            ax.set_xlabel('Episode time [s]')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f'Plan selected at t={time_grid[sample]:.4g}s: later plant controls are replanned; only first interval shares the same action')
        fig.savefig(out / 'selected_prediction_horizon.png', dpi=150)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, layout='constrained')
    for j in range(2):
        axes[0, j].step(time_grid, np.r_[commands[:, j], commands[-1, j]], where='post', label='Applied MPC command')
        axes[0, j].axhline(low[j], color='red', linestyle=':')
        axes[0, j].axhline(high[j], color='red', linestyle=':')
        if recorded_commands is not None:
            axes[0, j].step(time_grid, np.r_[recorded_commands[:, j], recorded_commands[-1, j]], where='post', linestyle='--', label='Recorded commands')
        axes[0, j].set_title(('Lift', 'Jib')[j] + ' command')
        axes[0, j].set_ylabel('V')
        axes[0, j].legend()
    for j, ax in enumerate(axes[1:].flat):
        ax.plot(time_grid, plant_states[:, j] - reference[:, j], label='Plant tracking error')
        ax.plot(time_grid, predicted[:, j] - plant_states[:, j], '--', label='Predictor minus plant')
        ax.set_title(MECHANICAL_COLS[j] + ' errors')
        ax.set_ylabel('rad' if j < 2 else 'rad/s')
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Episode time [s]')
        ax.grid(alpha=0.3)
    fig.savefig(out / 'commands_tracking_mismatch.png', dpi=150)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout='constrained')
    axes[0].plot(time_grid[:-1], solve_times * 1000, label='Optimizer solve time')
    axes[0].axhline(dt_control * 1000, color='red', linestyle='--', label='Requested control interval')
    axes[0].set_ylabel('ms')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_grid[:-1], costs, label='Full-model horizon objective')
    axes[1].set_ylabel('Normalized weighted cost')
    axes[1].set_xlabel('Episode time [s]')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(out / 'solve_time_and_cost.png', dpi=150)
    make_boom_plots(out, time_grid, reference, plant_states, predicted, commands, geometry, desired_tip)
    if show:
        print('Close ALL plot windows to finish the script.', flush=True)
        plt.show(block=True)
    plt.close('all')

def show_saved_results(folder, show=True, geometry=None):
    folder = Path(folder).resolve()
    config = read_report(folder / 'config.json')
    if geometry is None and config.get('geometry'):
        geometry = PlanarGeometry(**config['geometry'])
    if geometry is None and (folder / 'kinematics_config.json').is_file():
        geometry = PlanarGeometry(**json.loads((folder / 'kinematics_config.json').read_text()))
    print((folder / 'results.txt').read_text(encoding='utf-8'), flush=True)
    with np.load(folder / 'mpc_mechanical_plant_arrays.npz') as a:
        desired_tip = a['reference_end_effector_physical'] if 'reference_end_effector_physical' in a.files else None
        if geometry is not None:
            save_kinematic_results(folder, a['time_seconds'], a['reference_physical'], a['plant_mechanical_physical'], a['predictor_one_interval_mechanical_physical'], geometry, desired_tip, config.get('tracking_space', 'joint'))
        make_plots(folder, a['time_seconds'], a['reference_physical'], a['plant_mechanical_physical'], a['predictor_one_interval_mechanical_physical'], a['applied_commands_physical'], a['solve_seconds'], a['cost'], config['control_interval_seconds'], np.asarray(config['command_bounds']['low']), np.asarray(config['command_bounds']['high']), show, a['predictor_horizons_mechanical_physical'], a['teacher_reference_horizons_physical'], a['recorded_commands_physical'] if 'recorded_commands_physical' in a.files else None, geometry, desired_tip)

def main():
    p = argparse.ArgumentParser(description='iMotion residual dynamics')
    add_geometry_arguments(p)
    p.add_argument('--plot-results', type=Path, help='Print and open plots from a saved run, without solving again')
    p.add_argument('--checkpoint', '--full-checkpoint', dest='checkpoint', type=Path)
    p.add_argument('--mechanical-checkpoint', type=Path, default=DEFAULT_PLANT_CHECKPOINT)
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument('--test-file', type=Path, default=Path('Simulation_Output_53.csv'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--overwrite', action='store_true', help='Archive and replace the named result folder')
    p.add_argument('--segment-from', type=Path, help='Use the same start row as an existing run')
    p.add_argument('--horizon-ms', type=float, help='Horizon duration in milliseconds: e.g.20,50,100,200')
    p.add_argument('--reference-mode', choices=['recorded', 'plant-generated'], default='recorded')
    p.add_argument('--allow-reference-mismatch', action='store_true', help='Explicitly continue with a recorded reference that fails reproduction')
    p.add_argument('--replay-only', action='store_true')
    p.add_argument('--reachability-angle-tolerance', type=float, default=0.002)
    p.add_argument('--reachability-velocity-tolerance', type=float, default=0.002)
    p.add_argument('--terminal-angle-weight', type=float)
    p.add_argument('--terminal-velocity-weight', type=float)
    p.add_argument('--tracking-scales', type=float, nargs=4, help='Physical angle/velocity error scales; default training std')
    p.add_argument('--duration', type=float, default=1.0, help='Requested reference episode seconds')
    p.add_argument('--start-seconds', type=float, help='Time within retained data; default selects joint mechanical motion')
    p.add_argument('--block-steps', type=int, default=100, help='100 internal 10us steps = 1ms held command')
    p.add_argument('--horizon', type=int, default=100, help='Prediction horizon in control intervals')
    p.add_argument('--moves', type=int, default=10, help='Move blocks: e.g.100ms horizon/10 moves gives10ms planned blocks')
    p.add_argument('--maxiter', type=int, default=15)
    p.add_argument('--fd-step', type=float, default=0.03, help='SLSQP finite-difference step in [-1,1] decision units, to avoid float32 quantization')
    p.add_argument('--ftol', type=float, default=1e-08)
    p.add_argument('--angle-weight', type=float)
    p.add_argument('--velocity-weight', type=float)
    p.add_argument('--effort-weight', type=float, default=0.0001)
    p.add_argument('--smoothness-weight', type=float, default=0.001)
    p.add_argument('--lift-bounds', type=float, nargs=2, metavar=('LOW', 'HIGH'))
    p.add_argument('--jib-bounds', type=float, nargs=2, metavar=('LOW', 'HIGH'))
    p.add_argument('--bound-samples-per-file', type=int, default=1000)
    p.add_argument('--neutral-command', type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument('--progress-every', type=int, default=25)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--no-show', action='store_true', help='Save plots without opening windows')
    a = p.parse_args()
    saved_geometry = read_report(a.plot_results / 'config.json').get('geometry') if a.plot_results else None
    geometry = geometry_from_args(a, p, saved_geometry)
    if a.plot_results is not None:
        show_saved_results(a.plot_results, show=not a.no_show, geometry=geometry)
        return
    if a.cartesian_reference is not None and a.tracking_space != 'cartesian':
        p.error('--cartesian-reference requires --tracking-space cartesian')
    joint_defaults = (10.0, 1.0, 20.0, 5.0) if a.tracking_space == 'joint' else (0.0, 0.0, 0.0, 0.0)
    for key, value in zip(('angle_weight', 'velocity_weight', 'terminal_angle_weight', 'terminal_velocity_weight'), joint_defaults):
        if getattr(a, key) is None:
            setattr(a, key, value)
    cartesian_weights = np.asarray([a.position_weight, a.cartesian_velocity_weight, a.terminal_position_weight, a.terminal_cartesian_velocity_weight], float)
    if not np.isfinite(cartesian_weights).all() or (cartesian_weights < 0).any() or (a.tracking_space == 'cartesian' and cartesian_weights[:2].sum() <= 0):
        p.error('Cartesian weights must be finite, nonnegative and enable position or velocity tracking')
    if not all(math.isfinite(value) and value > 0 for value in (a.position_scale, a.cartesian_velocity_scale)):
        p.error('Cartesian error scales must be positive metres and m/s')
    if min(a.block_steps, a.horizon, a.moves, a.maxiter, a.bound_samples_per_file, a.progress_every, a.threads) < 1:
        p.error('Counts must be positive')
    if a.moves > a.horizon:
        p.error('moves must not exceed horizon')
    if not all((math.isfinite(value) and value > 0 for value in (a.duration, a.fd_step, a.ftol))) or a.fd_step > 1:
        p.error('duration/ftol must be positive; fd-step must be in (0,1]')
    if a.start_seconds is not None and (not math.isfinite(a.start_seconds) or a.start_seconds < 0):
        p.error('Invalid start-seconds')
    cost_weights = np.asarray([a.angle_weight, a.velocity_weight, a.effort_weight, a.smoothness_weight], float)
    if not np.isfinite(cost_weights).all() or (cost_weights < 0).any() or (a.tracking_space == 'joint' and a.angle_weight + a.velocity_weight <= 0):
        p.error('Cost weights must be nonnegative with positive tracking weight')
    if not all((math.isfinite(v) and v >= 0 for v in (a.terminal_angle_weight, a.terminal_velocity_weight))):
        p.error('Terminal weights must be finite and nonnegative')
    if not all((math.isfinite(v) and v > 0 for v in (a.reachability_angle_tolerance, a.reachability_velocity_tolerance))):
        p.error('Reproduction tolerances must be finite and positive')
    if a.tracking_scales is not None and (not all((math.isfinite(v) and v > 0 for v in a.tracking_scales))):
        p.error('Four positive physical tracking scales required')
    saved_segment = read_report(a.segment_from / 'config.json') if a.segment_from else None
    torch.set_num_threads(a.threads)
    checkpoint = (a.checkpoint or default_checkpoint()).resolve()
    trainer, model, ck = load_surrogate(checkpoint)
    weights = trainer.export_weights(model)
    sc = ck['scaler']
    require_completed(checkpoint, ck, p, 'Full-state predictor')
    plant_checkpoint = a.mechanical_checkpoint.resolve()
    if not plant_checkpoint.is_file():
        p.error('Mechanical plant checkpoint missing; train Mechanical States Model/trainMechanicalCompact.py first or supply --mechanical-checkpoint')
    plant = MechanicalPlant(plant_checkpoint)
    require_completed(plant_checkpoint, plant.checkpoint, p, 'Mechanical plant')
    if PLANT_INPUT_COLS != list(MECHANICAL_COLS) + [INPUT_COLS[i] for i in CI]:
        p.error('Mechanical plant signal order differs')
    if sc['input_columns'] != INPUT_COLS or sc['output_columns'] != OUTPUT_COLS:
        raise ValueError('Feature ordering differs')
    mean = np.asarray(sc['mean'], float)
    std = np.asarray(sc['std'], float)
    dt = float(sc['sampling_interval_seconds'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid model sampling interval')
    if not math.isclose(dt, plant.dt, rel_tol=1e-09, abs_tol=1e-15):
        p.error('Predictor and plant must have the same internal sampling interval')
    control_dt = a.block_steps * dt
    if a.horizon_ms is not None:
        if not math.isfinite(a.horizon_ms) or a.horizon_ms <= 0:
            p.error('horizon-ms must be positive')
        count = a.horizon_ms / (control_dt * 1000)
        if not math.isclose(count, round(count), abs_tol=1e-07):
            p.error('Horizon milliseconds must be an integer number of control intervals')
        a.horizon = int(round(count))
        if a.horizon < a.moves:
            p.error('Horizon must cover all move blocks')
    episode_steps = int(round(a.duration / control_dt))
    if episode_steps < 1:
        p.error('duration must contain at least one control interval')
    actual_duration = episode_steps * control_dt
    source = (a.test_file if a.test_file.is_absolute() else RAW_DATA_DIR / 'Test' / a.test_file).resolve()
    if any((same_recording(e['source_csv'], source) for e in ck['training_config']['training_manifest'])):
        p.error('Use a held-out test recording')
    if any((same_recording(f, source) for f in sc.get('fitted_files', []))):
        p.error('Reference recording contributed to scaler fitting')
    for entry in plant.checkpoint['training_config']['training_manifest']:
        if same_recording(entry['source_csv'], source):
            p.error('Reference must be held out from mechanical training too')
    if any((same_recording(path, source) for path in plant.checkpoint['scaler'].get('fitted_files', []))):
        p.error('Reference contributed to mechanical-plant normalization')
    if saved_segment:
        if Path(saved_segment['source_csv']).resolve() != source:
            p.error('Saved segment recording differs')
        if not math.isclose(saved_segment['dt_seconds'], dt, rel_tol=1e-09):
            p.error('Saved segment sampling differs')
        a.start_seconds = saved_segment['selected_retained_row'] * dt
    out = (a.output_dir or SCRIPT_DIR / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')).resolve()
    from reference_replay import prepare_output, report_replay, replay
    out = prepare_output(out, a.overwrite)
    x, cache = load_test(source, a.dataset_dir, sc, out)
    if a.lift_bounds is None or a.jib_bounds is None:
        low, high = sampled_training_bounds(ck, mean, std, a.bound_samples_per_file, a.dataset_dir.resolve())
    else:
        low = np.empty(2)
        high = np.empty(2)
    if a.lift_bounds is not None:
        low[0], high[0] = a.lift_bounds
    if a.jib_bounds is not None:
        low[1], high[1] = a.jib_bounds
    if not np.isfinite(low).all() or not np.isfinite(high).all() or (low >= high).any():
        p.error('Both command bounds must be finite LOW < HIGH')
    center = (low + high) / 2
    halfwidth = (high - low) / 2
    neutral = np.asarray(a.neutral_command, float)
    if not np.isfinite(neutral).all():
        p.error('Invalid neutral command')
    neutral = np.clip(neutral, low, high)
    selection = select_segment(x, episode_steps, a.block_steps, a.horizon, dt, a.start_seconds)
    start = activity_segment(x, selection, mean, std, episode_steps, a.block_steps)
    rows = start + np.arange(episode_steps + a.horizon + 1, dtype=np.int64) * a.block_steps
    reference_normalized = np.asarray(x[rows][:, MECH_INPUT], float)
    reference = reference_normalized * std[MECH_INPUT] + mean[MECH_INPUT]
    offset = float(sc['startup_rows_removed']) * dt
    csv_start = offset + start * dt
    assignment = np.arange(a.horizon, dtype=np.int64) * a.moves // a.horizon
    previous = np.array(x[start - 1], copy=True)
    current = np.array(x[start], copy=True)
    memory = trainer.native_encode(current, weights[0], weights[1])
    initial_full = np.stack((previous, current)).astype(float) * std + mean
    plant_columns = [INPUT_COLS.index(col) for col in PLANT_INPUT_COLS]
    plant.reset(initial_full[:, plant_columns])
    write_json(out / 'config.json', dict(source_csv=str(source), selected_retained_row=start, dt_seconds=dt, simulation='reproduction-first two-surrogate MPC'))
    replay_report, replay_reference, replay_exact, replay_held, recorded_commands, initial_plant_snapshot = report_replay(out / 'reference_reproduction', plant, x, start, episode_steps, a.block_steps, mean, std, a.reachability_angle_tolerance, a.reachability_velocity_tolerance, False, low, high)
    command_stats = pd.DataFrame(dict(command=['lift', 'jib'], recorded_mean_V=recorded_commands.mean(0), recorded_min_V=recorded_commands.min(0), recorded_max_V=recorded_commands.max(0)))
    print('Recorded commands for this exact segment:')
    print(command_stats.to_string(index=False))
    command_stats.to_csv(out / 'recorded_command_statistics.csv', index=False)
    if a.replay_only or (a.cartesian_reference is None and a.reference_mode == 'recorded' and (not replay_report['reproduction_gate_pass']) and (not a.allow_reference_mismatch)):
        message = 'Recorded-command reproduction failed; MPC was not run. This does not prove the trajectory is unreachable. Validate the plant/history, or explicitly choose --reference-mode plant-generated.'
        if a.replay_only:
            message = 'Replay-only diagnostic complete; no optimizer was run.'
        print(message, flush=True)
        (out / 'results.txt').write_text(message, encoding='utf-8')
        write_json(out / 'summary.json', dict(status='replay_only' if a.replay_only else 'recorded_reproduction_failed', mpc_executed=False, reproduction=replay_report))
        if not a.no_show:
            report_replay(out / 'reference_reproduction', plant, x, start, episode_steps, a.block_steps, mean, std, a.reachability_angle_tolerance, a.reachability_velocity_tolerance, True, low, high)
        return
    if a.reference_mode == 'plant-generated':
        _, _, reference, _, _ = replay(plant, x, start, episode_steps + a.horizon, a.block_steps, mean, std)
        reference_normalized = (reference - mean[MECH_INPUT]) / std[MECH_INPUT]
        print('Using explicit plant-generated held-command reference: a known input sequence produces this target.', flush=True)
    tip_reference = None
    if geometry is not None:
        tip_reference = geometry.end_effector(reference)
        if a.cartesian_reference:
            tip_reference = read_cartesian_reference(a.cartesian_reference, np.arange(len(reference)) * control_dt)
            radius = np.linalg.norm(tip_reference[:, :2] - [geometry.base_x, geometry.base_z], axis=1)
            if np.any(radius > geometry.lift_length + geometry.jib_length + 1e-9) or np.any(radius < abs(geometry.lift_length - geometry.jib_length) - 1e-9):
                p.error('Cartesian reference contains points outside the two-link geometric workspace')
        print(f'Tracking space={a.tracking_space}; planar geometry={asdict(geometry)}', flush=True)
        print(f'Cartesian normalization: position={a.position_scale:g} m; velocity={a.cartesian_velocity_scale:g} m/s', flush=True)
    plant.restore(initial_plant_snapshot)
    previous_command = current[CI].astype(float) * std[CI] + mean[CI]
    rollout = make_blocked_rollout(trainer)
    predictor_name = getattr(trainer, 'ARCHITECTURE', 'compact_latent')
    print(f"Model={predictor_name}; checkpoint epoch={ck['epoch']}; parameters={sum((p.numel() for p in model.parameters())):,}", flush=True)
    print(f'CSV segment={csv_start:.6f}..{csv_start + actual_duration:.6f}s; duration={actual_duration:g}s; internal dt={dt * 1000000.0:g}us; control interval={control_dt * 1000.0:g}ms; horizon={a.horizon * control_dt * 1000.0:g}ms; decision blocks={a.moves} ({2 * a.moves} variables)', flush=True)
    print(f'Bounds [V]: lift={low[0]:.6g}..{high[0]:.6g}, jib={low[1]:.6g}..{high[1]:.6g}; neutral={neutral.tolist()}', flush=True)
    print(f"Mechanical plant: epoch={plant.checkpoint['epoch']}; training epochs={plant.checkpoint['training_config']['arguments']['epochs']}; parameters={sum((value.numel() for value in plant.model.parameters()))}", flush=True)
    print('Offline two-surrogate experiment: only the full predictor is inside the optimizer. The mechanical plant advances after action selection; its feedback corrects only mechanics. Optimizer timing is measured, not assumed to meet the requested control interval.', flush=True)
    print('Compiling/warming blocked prediction kernel (excluded from solve timings)...', flush=True)
    compile_start = time.perf_counter()
    warm = np.tile(previous_command, (a.horizon, 1))
    rollout(previous, current, memory, warm, a.block_steps, mean, std, weights)
    warm_plant_kernel(plant.previous, plant.current, plant.latent, previous_command, 0, plant.mean, plant.std, *plant.weights)
    print(f'Kernel warmup={time.perf_counter() - compile_start:.2f}s', flush=True)
    config = dict(model=predictor_name, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), source_csv=str(source), cache=str(cache), selected_retained_row=start, selected_csv_time=csv_start, simulation='full-state optimizer predictor; separately trained mechanical Compact plant; no future recorded-state refresh', mechanical_checkpoint=str(plant_checkpoint), mechanical_checkpoint_sha256=hashlib.sha256(plant_checkpoint.read_bytes()).hexdigest(), feedback_policy='correct four mechanical fields in last two internal rows; preserve all other state estimates and full memory', initial_nonmechanical_state='seeded from two initial recording rows; subsequently predicted internally', strict_optimizer_plant_separation=True, reference_mode=a.reference_mode, reproduction=replay_report, terminal_weights=dict(angle=a.terminal_angle_weight, velocity=a.terminal_velocity_weight), preprocessing='same prepared cache; recorded commands use centered offline smoothing; optimized commands are held, not smoothed', dt_seconds=dt, control_interval_seconds=control_dt, actual_duration_seconds=actual_duration, command_bounds=dict(low=low.tolist(), high=high.tolist(), source='explicit bounds or sampled training commands'), normalized_mechanical_error_scale=(std[MECH_INPUT] if a.tracking_scales is None else np.asarray(a.tracking_scales)).tolist(), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    write_json(out / 'config.json', config)
    config.update(tracking_space=a.tracking_space, geometry=asdict(geometry) if geometry else None, cartesian_cost_weights=cartesian_weights.tolist(), cartesian_error_scales=dict(position_m=a.position_scale, velocity_m_s=a.cartesian_velocity_scale), cartesian_reference_source=str(a.cartesian_reference.resolve()) if a.cartesian_reference else 'forward kinematics of mechanical reference', joint_reference_role='replay guide, not inverse kinematics of external tip target' if a.cartesian_reference else 'mechanical trajectory reference')
    config['activity_selection'] = dict(mode='automatic' if a.start_seconds is None else 'explicit_start', velocity_units='rad/s', physical_reconstruction='normalized * std + mean', score='min(lift_RMS,jib_RMS) + 0.1*(lift_RMS+jib_RMS)')
    if a.cartesian_reference:
        config['reference_mode'] = 'external-cartesian'
    write_json(out / 'config.json', config)
    (out / 'mpc_surrogate_mechanical_plant_source.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    state_trace = np.empty((episode_steps + 1, 24), float)
    state_trace[0] = current.astype(float) * std + mean
    plant_trace = np.empty((episode_steps + 1, 4))
    plant_trace[0] = plant.state()
    predictor_trace = np.empty((episode_steps + 1, 4))
    predictor_trace[0] = state_trace[0, MECH_INPUT]
    predictor_prior_full = np.empty((episode_steps + 1, 24))
    predictor_prior_full[0] = state_trace[0]
    horizon_forecasts = np.empty((episode_steps, a.horizon, 4))
    horizon_full_states = np.empty((episode_steps, a.horizon, 22))
    plant_advance_times = np.empty(episode_steps)
    commands = np.empty((episode_steps, 2))
    solve_times = np.empty(episode_steps)
    costs = np.empty(episode_steps)
    planned_commands = np.empty((episode_steps, a.horizon, 2))
    teacher_states = np.empty((episode_steps, 24))
    teacher_refs = np.empty((episode_steps, a.horizon, 4))
    diagnostics = []
    teacher = []
    seed = np.tile(np.clip((previous_command - center) / halfwidth, -1, 1), (a.moves, 1)).ravel()
    started = time.perf_counter()
    last = time.perf_counter()
    for k in range(episode_steps):
        goal = reference_normalized[k + 1:k + 1 + a.horizon]
        obj = MPCObjective(rollout, previous, current, memory, goal, previous_command, center, halfwidth, neutral, assignment, a.block_steps, mean, std, weights, cost_weights, a.terminal_angle_weight, a.terminal_velocity_weight, a.tracking_scales, geometry if a.tracking_space == 'cartesian' else None, tip_reference[k + 1:k + 1 + a.horizon] if tip_reference is not None else None, cartesian_weights, [a.position_scale, a.position_scale, a.cartesian_velocity_scale, a.cartesian_velocity_scale])
        solve_start = time.perf_counter()
        hold = np.tile(np.clip((previous_command - center) / halfwidth, -1, 1), (a.moves, 1)).ravel()
        hold_cost = obj(hold)
        seed_cost = obj(seed)
        result = minimize(obj, seed, method='SLSQP', bounds=[(-1.0, 1.0)] * (2 * a.moves), options=dict(maxiter=a.maxiter, ftol=a.ftol, eps=a.fd_step, disp=False))
        obj(np.clip(result.x, -1, 1))
        if obj.best_decision is None:
            raise RuntimeError(f'No finite bounded rollout at control step {k}')
        best_plan = expand_moves(obj.best_decision, assignment, center, halfwidth)
        action = best_plan[0]
        forecast, _, _, _ = rollout(previous, current, memory, best_plan, a.block_steps, mean, std, weights)
        horizon_full_states[k] = forecast.astype(float) * std[SI] + mean[SI]
        horizon_forecasts[k] = horizon_full_states[k][:, MECH_OUTPUT]
        solve_times[k] = time.perf_counter() - solve_start
        teacher_states[k] = current.astype(float) * std + mean
        teacher_refs[k] = reference[k + 1:k + 1 + a.horizon]
        row = dict(step=k, time_s=k * control_dt, csv_time_s=csv_start + k * control_dt, previous_lift_V=previous_command[0], previous_jib_V=previous_command[1], optimal_lift_V=action[0], optimal_jib_V=action[1], solver_success=bool(result.success))
        for j, col in enumerate(MECHANICAL_COLS):
            row[f'state_{col}'] = plant_trace[k, j]
            row[f'reference_next_{col}'] = reference[k + 1, j]
        teacher.append(row)
        commit_start = time.perf_counter()
        previous, current, memory, predicted_mechanics, actual_mechanics, prior_full = advance_and_correct(rollout, previous, current, memory, action, a.block_steps, mean, std, weights, plant)
        plant_advance_times[k] = time.perf_counter() - commit_start
        predictor_trace[k + 1] = predicted_mechanics
        plant_trace[k + 1] = actual_mechanics
        predictor_prior_full[k + 1] = prior_full
        state_trace[k + 1] = current.astype(float) * std + mean
        commands[k] = action
        planned_commands[k] = best_plan
        costs[k] = obj.best_cost
        components = obj.best_components
        diagnostics.append(dict(step=k, time_s=k * control_dt, solve_seconds=solve_times[k], selected_cost=obj.best_cost, warm_start_cost=seed_cost, hold_command_cost=hold_cost, angle_error_term=components[0], velocity_error_term=components[1], effort_term=components[2], smoothness_term=components[3], terminal_angle_term=components[4], terminal_velocity_term=components[5], solver_success=bool(result.success), status=int(result.status), message=str(result.message), iterations=int(result.nit), candidate_calls=obj.calls, invalid_candidates=obj.invalid_calls))
        if geometry is not None and a.tracking_space == 'cartesian':
            diagnostics[-1].update(position_error_term=components[6], cartesian_velocity_error_term=components[7], terminal_position_term=components[8], terminal_cartesian_velocity_term=components[9])
        shifted = np.vstack((best_plan[1:], best_plan[-1]))
        starts = np.asarray([np.flatnonzero(assignment == i)[0] for i in range(a.moves)])
        seed = np.clip((shifted[starts] - center) / halfwidth, -1, 1).ravel()
        previous_command = action.copy()
        now = time.perf_counter()
        if (k + 1) % a.progress_every == 0 or now - last > 5 or k + 1 == episode_steps:
            elapsed = now - started
            print(f'MPC {k + 1}/{episode_steps}: elapsed={elapsed:.1f}s; ETA={elapsed * (episode_steps - k - 1) / (k + 1):.1f}s; solve={solve_times[k] * 1000.0:.2f}ms; cost={costs[k]:.6g}; u={action.tolist()}; optimizer={result.message}', flush=True)
            last = now
            pd.DataFrame(diagnostics).to_csv(out / 'solver_history.csv', index=False)
    time_grid = np.arange(episode_steps + 1) * control_dt
    predicted = plant_trace
    target = reference[:episode_steps + 1]
    metrics = tracking_metrics(target[1:], predicted[1:])
    metrics.to_csv(out / 'tracking_metrics.csv', index=False)
    mismatch_metrics = tracking_metrics(plant_trace[1:], predictor_trace[1:])
    mismatch_metrics.to_csv(out / 'model_mismatch_metrics.csv', index=False)
    history = pd.DataFrame(diagnostics)
    history.to_csv(out / 'solver_history.csv', index=False)
    pd.DataFrame(teacher).to_csv(out / 'teacher_dataset.csv', index=False)
    trace = pd.DataFrame(state_trace, columns=INPUT_COLS)
    trace.insert(0, 'time_s', time_grid)
    trace.insert(1, 'csv_time_s', csv_start + time_grid)
    for j, col in enumerate(MECHANICAL_COLS):
        trace[f'reference_{col}'] = target[:, j]
        trace[f'plant_{col}'] = plant_trace[:, j]
        trace[f'predictor_prior_{col}'] = predictor_trace[:, j]
        trace[f'tracking_error_{col}'] = plant_trace[:, j] - target[:, j]
        trace[f'model_mismatch_{col}'] = predictor_trace[:, j] - plant_trace[:, j]
    trace.to_csv(out / 'trajectory.csv', index=False)
    pd.DataFrame(dict(time_s=time_grid[:-1], lift_command_V=commands[:, 0], jib_command_V=commands[:, 1])).to_csv(out / 'commands.csv', index=False)
    np.savez(out / 'mpc_mechanical_plant_arrays.npz', controller_state_estimate_physical=state_trace, reference_physical=target, plant_mechanical_physical=plant_trace, predictor_one_interval_mechanical_physical=predictor_trace, predictor_prior_full_state_physical=predictor_prior_full, predictor_horizons_mechanical_physical=horizon_forecasts, predictor_horizons_full_state_physical=horizon_full_states, model_mismatch_physical=predictor_trace - plant_trace, commit_seconds=plant_advance_times, teacher_full_state_estimate_physical=teacher_states, teacher_reference_horizons_physical=teacher_refs, selected_command_sequences_physical=planned_commands, applied_commands_physical=commands, solve_seconds=solve_times, cost=costs, time_seconds=time_grid, recorded_commands_physical=recorded_commands, recorded_reference_physical=replay_reference, recorded_exact_replay_physical=replay_exact, recorded_held_replay_physical=replay_held)
    if geometry is not None:
        kinematic_summary = save_kinematic_results(out, time_grid, target, plant_trace, predictor_trace, geometry, tip_reference[:episode_steps + 1], a.tracking_space)
        with np.load(out / 'mpc_mechanical_plant_arrays.npz') as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays.update(reference_end_effector_physical=tip_reference[:episode_steps + 1], predictor_horizons_end_effector_physical=geometry.end_effector(horizon_forecasts), teacher_end_effector_horizons_physical=np.stack([tip_reference[k + 1:k + 1 + a.horizon] for k in range(episode_steps)]))
        np.savez(out / 'mpc_mechanical_plant_arrays.npz', **arrays)
    comparison = pd.DataFrame(dict(time_s=time_grid[:-1], mpc_lift_V=commands[:, 0], recorded_lift_V=recorded_commands[:, 0], mpc_jib_V=commands[:, 1], recorded_jib_V=recorded_commands[:, 1]))
    comparison.to_csv(out / 'mpc_vs_recorded_commands.csv', index=False)
    command_delta = commands - recorded_commands
    saturation = np.mean(np.isclose(commands, low, atol=1e-05) | np.isclose(commands, high, atol=1e-05), axis=0) * 100
    summary = dict(mpc_executed=True, reference_mode=a.reference_mode, reproduction=replay_report, command_difference_rmse_V=np.sqrt(np.mean(command_delta ** 2, 0)).tolist(), control_steps=episode_steps, control_interval_seconds=control_dt, horizon_seconds=a.horizon * control_dt, mean_solve_ms=float(solve_times.mean() * 1000), median_solve_ms=float(np.median(solve_times) * 1000), p95_solve_ms=float(np.percentile(solve_times, 95) * 1000), max_solve_ms=float(solve_times.max() * 1000), deadline_miss_percent=float(np.mean(solve_times > control_dt) * 100), solver_success_percent=float(history.solver_success.mean() * 100), lift_saturation_percent=float(saturation[0]), jib_saturation_percent=float(saturation[1]), improved_over_hold_percent=float(np.mean(history.selected_cost < history.hold_command_cost - 1e-10) * 100), tracking_metrics=json.loads(metrics.to_json(orient='records')), mismatch_metrics=json.loads(mismatch_metrics.to_json(orient='records')), mechanical_plant_advances=episode_steps, optimizer_uses_mechanical_plant=False, mean_commit_ms=float(plant_advance_times.mean() * 1000), simulated_plant='separate mechanical learned model; learned-model mismatch experiment, not physical crane validation')
    write_json(out / 'summary.json', summary)
    summary['tracking_space'] = a.tracking_space
    if a.cartesian_reference:
        summary['reference_mode'] = 'external-cartesian'
    if geometry is not None:
        summary['kinematic_tracking'] = kinematic_summary
    write_json(out / 'summary.json', summary)
    metric_title = 'Joint guide deviations (external Cartesian target)' if a.cartesian_reference else 'Joint tracking metrics'
    text = metric_title + ' (physical units; initial measured sample excluded):\n' + metrics.to_string(index=False, float_format=lambda v: f'{v:.8g}')
    text += '\n\nOne-control-interval predictor-minus-plant mismatch:\n' + mismatch_metrics.to_string(index=False, float_format=lambda v: f'{v:.8g}')
    text += '\n\nControl/solver summary:\n' + json.dumps(summary, indent=2) + f'\n\nSaved results: {out}'
    if geometry is not None:
        text += '\n\n' + (out / 'kinematics_results.txt').read_text(encoding='utf-8')
    print('\n' + text, flush=True)
    (out / 'results.txt').write_text(text, encoding='utf-8')
    make_plots(out, time_grid, target, plant_trace, predictor_trace, commands, solve_times, costs, control_dt, low, high, not a.no_show, horizon_forecasts, teacher_refs, recorded_commands, geometry, tip_reference[:episode_steps + 1] if tip_reference is not None else None)
if __name__ == '__main__':
    main()
