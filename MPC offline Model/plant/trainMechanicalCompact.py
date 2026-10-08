import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, resolve_array, same_recording
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import math
import time
import shutil
import numpy as np
import torch
from mechanical_compact import SCRIPT_DIR, DEFAULT_DATASET_DIR, DEFAULT_ARTIFACT_DIR, FORMAT, INPUT_COLS, OUTPUT_COLS, STATE_INDICES, COMMAND_INDICES, STATE_ARRAY, ModelConfig, MechanicalCompact, load_training_records, write_json, export_weights, native_encode, native_chunk
ANGLE_COLS = tuple(OUTPUT_COLS[:2])
VELOCITY_COLS = tuple(OUTPUT_COLS[2:])

def batch(records, pairs, horizon):
    return torch.from_numpy(np.stack([r['array'][s - 1:s + horizon + 1] for r, s in ((records[i], s) for i, s in pairs)]).copy())

def state_loss_weights(angle_weight=4.0, velocity_weight=4.0):
    weights = np.ones(4, dtype=np.float32)
    for col in ANGLE_COLS:
        weights[OUTPUT_COLS.index(col)] = angle_weight
    for col in VELOCITY_COLS:
        weights[OUTPUT_COLS.index(col)] = velocity_weight
    return weights

def weighted_mse(error, weights):
    return (error.square() * weights).sum(dim=-1).mean() / weights.sum()

def sample_horizon(rng, short, medium, long, medium_probability=0.6, long_probability=0.2):
    choice = rng.random()
    if choice < medium_probability:
        return medium
    if choice < medium_probability + long_probability:
        return long
    return short

def train_batch(model, optimizer, rows, block, initial_latent=None, loss_weights=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    previous, current = (rows[:, 0], rows[:, 1])
    z = model.encode(current) if initial_latent is None else initial_latent
    horizon = rows.shape[1] - 2
    total = 0.0
    loss_scale = model.scale.clamp_min(0.01)
    weights = rows.new_ones(4) if loss_weights is None else torch.as_tensor(loss_weights, dtype=rows.dtype, device=rows.device)
    for start in range(0, horizon, block):
        loss = rows.new_zeros(())
        for t in range(start, min(start + block, horizon)):
            state, z = model.step(previous, current, z)
            loss += weighted_mse((state - rows[:, t + 2, STATE_INDICES]) / loss_scale, weights) / horizon
            following = torch.empty_like(current)
            following[:, COMMAND_INDICES] = rows[:, t + 2, COMMAND_INDICES]
            following[:, STATE_INDICES] = state
            previous, current = (current, following)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        total += float(loss.detach())
        previous, current, z = (previous.detach(), current.detach(), z.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total

def native_validation(model, rows, horizons, loss_weights=None, return_weighted=False):
    weights = export_weights(model)
    mean = np.zeros(6)
    std = np.ones(6)
    scores = {str(h): 0.0 for h in horizons}
    weighted = {str(h): 0.0 for h in horizons}
    state_weights = np.ones(4) if loss_weights is None else np.asarray(loss_weights, dtype=float)
    for window in rows:
        z = native_encode(window[1], weights[0], weights[1])
        prediction, _, _, _ = native_chunk(window[1:-1], window[0], window[1], z, True, mean, std, *weights)
        for h in horizons:
            errors = (prediction[:h] - window[2:2 + h, STATE_ARRAY]) ** 2
            scores[str(h)] += float(np.mean(errors)) / len(rows)
            weighted[str(h)] += float(np.mean(np.sum(errors * state_weights, axis=1) / state_weights.sum())) / len(rows)
    return (scores, weighted) if return_weighted else scores

def burn_in_batch(model, rows, steps):
    if not steps:
        return (rows, None)
    weights = export_weights(model)
    mean = np.zeros(6)
    std = np.ones(6)
    suffix = rows[:, steps:].clone()
    latents = []
    for i, window in enumerate(rows.numpy()):
        z = native_encode(window[1], weights[0], weights[1])
        _, previous, current, z = native_chunk(window[1:steps + 1], window[0], window[1], z, True, mean, std, *weights)
        current[list(COMMAND_INDICES)] = window[steps + 1, list(COMMAND_INDICES)]
        suffix[i, 0] = torch.from_numpy(previous)
        suffix[i, 1] = torch.from_numpy(current)
        latents.append(z)
    return (suffix, torch.from_numpy(np.stack(latents)))

def main():
    p = argparse.ArgumentParser(description='iMotion residual dynamics')
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--overwrite', action='store_true', help='Replace existing checkpoints and history with a fresh training run')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--steps-per-epoch', type=int, default=20)
    p.add_argument('--hidden', type=int, default=32, help='Decoder hidden width; recurrent memory size is --latent-dim')
    p.add_argument('--latent-dim', type=int, default=8, help='Compact tanh recurrent memory width')
    p.add_argument('--angle-weight', type=float, default=4.0, help='Loss weight for lift/jib angles; other states use 1')
    p.add_argument('--velocity-weight', type=float, default=4.0, help='Loss weight for lift/jib angular velocities')
    p.add_argument('--medium-horizon', type=int, default=500, help='Medium rollout steps at dataset sample interval')
    p.add_argument('--medium-probability', type=float, default=0.6)
    p.add_argument('--long-probability', type=float, default=0.2)
    p.add_argument('--selection-horizon', type=int, help='Select best checkpoint on weighted error at this horizon; default medium horizon')
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--short-horizon', type=int, default=100)
    p.add_argument('--long-horizon', type=int, default=1000)
    p.add_argument('--burn-in-steps', type=int, default=10000, help='Maximum unrefreshed prefix before differentiable training suffix')
    p.add_argument('--grad-block', type=int, default=50)
    p.add_argument('--val-horizons', type=int, nargs='+', default=[100, 500, 1000, 2000, 10000, 100000])
    p.add_argument('--val-windows', type=int, default=4)
    p.add_argument('--scale-samples-per-file', type=int, default=10000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.selection_horizon = a.medium_horizon if a.selection_horizon is None else a.selection_horizon
    a.val_horizons = sorted(set(a.val_horizons + [a.selection_horizon]))
    if min(a.epochs, a.batch_size, a.steps_per_epoch, a.hidden, a.latent_dim, a.short_horizon, a.medium_horizon, a.long_horizon, a.selection_horizon, a.grad_block, a.val_windows, a.threads, a.scale_samples_per_file, *a.val_horizons) < 1 or min(a.warmup_epochs, a.burn_in_steps) < 0:
        p.error('Invalid positive counts')
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0 or a.short_horizon > a.long_horizon:
        p.error('Invalid learning rate/horizons')
    if not a.short_horizon <= a.medium_horizon <= a.long_horizon:
        p.error('Require short <= medium <= long horizons')
    if not all((math.isfinite(v) and v > 0 for v in (a.angle_weight, a.velocity_weight))):
        p.error('Loss weights must be finite and positive')
    if not all((math.isfinite(v) and 0 <= v <= 1 for v in (a.medium_probability, a.long_probability))) or a.medium_probability + a.long_probability > 1:
        p.error('Horizon probabilities must be in [0,1] and sum to at most 1')
    if a.epochs <= a.warmup_epochs:
        p.error('epochs must exceed warmup-epochs so recursive curriculum is trained')
    if not a.overwrite and a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output folder is nonempty; choose a new --output-dir or use --overwrite')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    records, sc, manifest = load_training_records(a.dataset_dir.resolve())
    sums = np.zeros(4)
    count = 0
    for r in records:
        indices = np.unique(np.linspace(1, r['train_rows'] - 1, min(r['train_rows'] - 1, a.scale_samples_per_file), dtype=int))
        if not len(indices):
            continue
        delta = np.asarray(r['array'][indices][:, STATE_INDICES], float) - r['array'][indices - 1][:, STATE_INDICES]
        sums += (delta ** 2).sum(axis=0)
        count += len(delta)
    if count == 0:
        raise ValueError('Insufficient training rows for delta scale')
    scale = np.clip(np.sqrt(sums / count), 1e-05, 1.0)
    max_train = max(5, a.long_horizon) + a.burn_in_steps
    max_val = max(a.val_horizons)
    counts = np.array([max(0, r['train_rows'] - max_train - 1) for r in records])
    if counts.sum() == 0:
        raise ValueError('No training windows')
    val_counts = [max(0, (r['rows'] - r['validation_start'] - 2 - max_val) // max_val + 1) for r in records]
    total_val = sum(val_counts)
    if not total_val:
        raise ValueError('No validation windows')
    cumulative = np.cumsum(val_counts)
    pairs = []
    for flat in np.unique(np.linspace(0, total_val - 1, min(total_val, a.val_windows), dtype=int)):
        i = int(np.searchsorted(cumulative, flat, side='right'))
        before = cumulative[i - 1] if i else 0
        pairs.append((i, records[i]['validation_start'] + 1 + int(flat - before) * max_val))
    cfg = ModelConfig(a.latent_dim, a.hidden)
    model = MechanicalCompact(cfg, scale)
    loss_weights = state_loss_weights(a.angle_weight, a.velocity_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-06)
    config = dict(format=FORMAT, model=asdict(cfg), training_manifest=manifest, scaler_fitted_rows=sc['fitted_rows'], input_columns=INPUT_COLS, output_columns=OUTPUT_COLS, feature_columns=INPUT_COLS + ['delta_' + col for col in INPUT_COLS], excludes_hydraulic_electrical_features=True, delta_scale_training_pairs=count, loss_weights=dict(zip(OUTPUT_COLS, loss_weights.tolist())), checkpoint_selection=f'weighted normalized MSE at {a.selection_horizon} steps', command_alignment='u[t],x[t],x[t-1] -> x[t+1]', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        previous = [a.output_dir / name for name in ('best_model.pt', 'latest_model.pt', 'best_numba_weights.npz', 'config.json', 'scaler.json', 'training_history.json', 'trainMechanicalCompact_source.py', 'mechanical_compact_source.py') if (a.output_dir / name).is_file()]
        if previous:
            backup = a.output_dir / 'previous_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            backup.mkdir(parents=True)
            for path in previous:
                shutil.copy2(path, backup / path.name)
            for path in previous:
                if path.parent.resolve() != a.output_dir.resolve():
                    raise ValueError('Overwrite target is outside output folder')
                path.unlink()
            print(f'Previous mechanical artifacts copied to: {backup}', flush=True)
    write_json(a.output_dir / 'config.json', config)
    write_json(a.output_dir / 'scaler.json', sc)
    shutil.copy2(Path(__file__).parent / 'mechanical_compact.py', a.output_dir / 'mechanical_compact_source.py')
    (a.output_dir / 'trainMechanicalCompact_source.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    dt = sc['sampling_interval_seconds']
    print(f'CPU training; parameters={sum((p.numel() for p in model.parameters())):,}; dt={dt * 1000000.0:g} us; validation horizons={[h * dt for h in a.val_horizons]} s; Numba inference exported after each epoch', flush=True)
    print(f'Curriculum after warmup: short={a.short_horizon * dt:g}s ({1 - a.medium_probability - a.long_probability:.0%}), medium={a.medium_horizon * dt:g}s ({a.medium_probability:.0%}), long={a.long_horizon * dt:g}s ({a.long_probability:.0%}); checkpoint selection={a.selection_horizon * dt:g}s weighted error', flush=True)
    validation_rows = batch(records, pairs, max_val).numpy()
    compile_started = time.perf_counter()
    native_validation(model, validation_rows[:1, :3], [1])
    print(f'Numba validation warmup: {time.perf_counter() - compile_started:.2f} s (excluded from epoch timings)', flush=True)
    best = math.inf
    history = []
    for epoch in range(1, a.epochs + 1):
        started = time.perf_counter()
        total = 0.0
        prefix_total = 0
        horizon_total = 0
        horizon_counts = {}
        for step in range(a.steps_per_epoch):
            h = 5 if epoch <= a.warmup_epochs else sample_horizon(rng, a.short_horizon, a.medium_horizon, a.long_horizon, a.medium_probability, a.long_probability)
            prefix = 0 if epoch <= a.warmup_epochs or rng.random() < 0.25 else int(rng.integers(0, a.burn_in_steps + 1))
            ids = rng.choice(len(records), size=a.batch_size, p=counts / counts.sum())
            selected = [(int(i), int(rng.integers(1, counts[i] + 1))) for i in ids]
            rows, z = burn_in_batch(model, batch(records, selected, h + prefix), prefix)
            total += train_batch(model, opt, rows, a.grad_block, z, loss_weights)
            prefix_total += prefix
            horizon_total += h
            horizon_counts[str(h)] = horizon_counts.get(str(h), 0) + 1
        training_seconds = time.perf_counter() - started
        model.eval()
        validation_started = time.perf_counter()
        scores, weighted_scores = native_validation(model, validation_rows, a.val_horizons, loss_weights, True)
        validation_seconds = time.perf_counter() - validation_started
        score = weighted_scores[str(a.selection_horizon)]
        if not all((math.isfinite(value) for value in list(scores.values()) + list(weighted_scores.values()))):
            raise RuntimeError('Nonfinite validation rollout')
        ck = dict(format=FORMAT, model_config=asdict(cfg), model=model.state_dict(), scale=scale.tolist(), scaler=sc, training_config=config, epoch=epoch, validation_loss=score, optimizer=opt.state_dict())
        for name in ['latest_model.pt', 'best_model.pt'] if score < best else ['latest_model.pt']:
            temp = a.output_dir / (name + '.tmp')
            torch.save(ck, temp)
            temp.replace(a.output_dir / name)
        if score < best:
            with (a.output_dir / 'best_numba_weights.npz.tmp').open('wb') as f:
                np.savez(f, **{f'w{i}': w for i, w in enumerate(export_weights(model))})
            (a.output_dir / 'best_numba_weights.npz.tmp').replace(a.output_dir / 'best_numba_weights.npz')
        best = min(best, score)
        row = dict(epoch=epoch, train_loss=total / a.steps_per_epoch, validation=scores, weighted_validation=weighted_scores, validation_loss=score, training_seconds=training_seconds, validation_seconds=validation_seconds, seconds=time.perf_counter() - started, mean_burn_in_steps=prefix_total / a.steps_per_epoch, mean_gradient_rollout_steps=horizon_total / a.steps_per_epoch, horizon_counts=horizon_counts)
        history.append(row)
        write_json(a.output_dir / 'training_history.json', history)
        print(f"Epoch {epoch}/{a.epochs}: train={row['train_loss']:.7g}; validation={scores}; best={best:.7g}; train_s={training_seconds:.2f}; validation_s={validation_seconds:.3f}; seconds={row['seconds']:.1f}", flush=True)
    print(f'Artifacts: {a.output_dir}', flush=True)
if __name__ == '__main__':
    main()
