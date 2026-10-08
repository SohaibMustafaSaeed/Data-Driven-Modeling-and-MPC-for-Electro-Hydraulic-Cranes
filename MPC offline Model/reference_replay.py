from common.project import ROOT
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

def prepare_output(folder, overwrite=False):
    folder = Path(folder).resolve()
    workspace = ROOT
    if folder.exists() and any(folder.iterdir()):
        if not overwrite:
            raise ValueError(f'Output is nonempty; use --overwrite: {folder}')
        if not folder.is_relative_to(workspace) or folder in (workspace, Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
            raise ValueError('Overwrite target must be a specific result folder within the workspace')
        if not (folder / 'config.json').is_file():
            raise ValueError('Overwrite requires an existing result config.json')
        info = json.loads((folder / 'config.json').read_text())
        if 'simulation' not in info or 'source_csv' not in info:
            raise ValueError('Overwrite accepts experiment results only, not trained model or project folders')
        backup = folder.with_name(folder.name + '_backup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        if not backup.resolve().is_relative_to(workspace):
            raise ValueError('Backup path is outside workspace')
        folder.rename(backup)
        print(f'Previous result folder archived: {backup}', flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def replay(plant, x, start, steps, block, full_mean, full_std):
    from offline_mpc import INPUT_COLS, MECH_INPUT, CI
    from mechanical_compact import INPUT_COLS as plant_cols, native_chunk
    initial = x[start - 1:start + 1].astype(float) * full_std + full_mean
    mapping = [INPUT_COLS.index(col) for col in plant_cols]
    plant.reset(initial[:, mapping])
    snapshot = plant.snapshot()
    exact = np.empty((steps + 1, 4))
    held = np.empty_like(exact)
    exact[0] = held[0] = plant.state()
    commands = np.empty((steps, 2))
    previous, current, memory = snapshot
    for k in range(steps):
        normalized = x[start + k * block:start + (k + 1) * block, CI]
        physical = normalized.astype(float) * full_std[CI] + full_mean[CI]
        commands[k] = physical[0]
        rows = np.zeros((block, 6), np.float32)
        rows[:, 4:] = (physical - plant.mean[4:]) / plant.std[4:]
        prediction, previous, current, memory = native_chunk(rows, previous, current, memory, True, plant.mean, plant.std, *plant.weights)
        exact[k + 1] = prediction[-1]
        held[k + 1] = plant.advance(commands[k], internal_steps=block)
    plant.restore(snapshot)
    indices = start + np.arange(steps + 1) * block
    reference = x[indices][:, MECH_INPUT].astype(float) * full_std[MECH_INPUT] + full_mean[MECH_INPUT]
    return (reference, exact, held, commands, snapshot)

def report_replay(out, plant, x, start, steps, block, mean, std, angle_tol=0.002, velocity_tol=0.002, show=False, low=None, high=None):
    from offline_mpc import tracking_metrics, MECHANICAL_COLS, write_json
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref, exact, held, commands, snapshot = replay(plant, x, start, steps, block, mean, std)
    tables = []
    for name, prediction in [('exact_prepared_commands', exact), ('held_recorded_commands', held)]:
        table = tracking_metrics(ref[1:], prediction[1:])
        table.insert(0, 'replay', name)
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(out / 'reproduction_metrics.csv', index=False)
    tolerance = np.array([angle_tol, angle_tol, velocity_tol, velocity_tol])
    exact_ok = bool(np.all(tables[0].RMSE.to_numpy() <= tolerance))
    held_ok = bool(np.all(tables[1].RMSE.to_numpy() <= tolerance))
    in_bounds = True if low is None else bool(np.all(commands >= low - 1e-06) and np.all(commands <= high + 1e-06))
    summary = dict(exact_reproduction_pass=exact_ok, held_reproduction_pass=held_ok, held_commands_in_mpc_bounds=in_bounds, reproduction_gate_pass=held_ok and in_bounds, angle_rmse_tolerance_rad=angle_tol, velocity_rmse_tolerance_rad_s=velocity_tol, interpretation='Configurable recorded-command reproduction test; failure does not prove mathematical unreachability.', command_source='same prepared centered-smoothed recorded commands as model preprocessing', steps=steps, block_steps=block, selected_retained_row=int(start), metrics=json.loads(metrics.to_json(orient='records')))
    write_json(out / 'summary.json', summary)
    np.savez(out / 'replay_arrays.npz', reference=ref, exact=exact, held=held, recorded_commands=commands, time=np.arange(steps + 1) * block * plant.dt)
    text = metrics.to_string(index=False, float_format=lambda value: f'{value:.8g}')
    print('Recorded-command reproduction (physical RMSE/MAE/max; R2 supplementary):', flush=True)
    print(text, flush=True)
    print(f"Reproduction gate: {('PASS' if summary['reproduction_gate_pass'] else 'FAIL')}", flush=True)
    (out / 'results.txt').write_text(text + chr(10) + json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('TkAgg' if show else 'Agg')
    import matplotlib.pyplot as plt
    t = np.arange(steps + 1) * block * plant.dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout='constrained')
    for j, ax in enumerate(axes.flat):
        ax.plot(t, ref[:, j], color='black', label='Recorded reference')
        ax.plot(t, exact[:, j], '--', label='Plant: exact prepared commands')
        ax.plot(t, held[:, j], ':', label='Plant: controller-rate held commands')
        ax.set_title(MECHANICAL_COLS[j])
        ax.set_ylabel('rad' if j < 2 else 'rad/s')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel('Episode time [s]')
    fig.savefig(out / 'recorded_reference_reproduction.png', dpi=150)
    if show:
        plt.show(block=True)
    plt.close(fig)
    return (summary, ref, exact, held, commands, snapshot)
