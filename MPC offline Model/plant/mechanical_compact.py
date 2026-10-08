import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, resolve_array, same_recording
from dataclasses import dataclass
import json
import numpy as np
import torch
from torch import nn
from numba import njit
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PREPARED_DIR
FULL_STATE_DIR = ROOT / 'common'
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR.parent / 'artifacts/mechanical_plant'
FORMAT = 'mechanical_compact_recurrent_numba_6x4_v1'
INPUT_COLS = ['thetalift[rad]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]', 'ulift[V]', 'ujib[V]']
OUTPUT_COLS = INPUT_COLS[:4]
STATE_INDICES = (0, 1, 2, 3)
COMMAND_INDICES = (4, 5)
STATE_ARRAY = np.asarray(STATE_INDICES, np.int64)

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def project_scaler(full):
    selected = [full['input_columns'].index(col) for col in INPUT_COLS]
    return dict(format='mechanical_only_training_scaler_v1', input_columns=INPUT_COLS, output_columns=OUTPUT_COLS, mean=np.asarray(full['mean'])[selected].tolist(), std=np.asarray(full['std'])[selected].tolist(), fitted_rows=full['fitted_rows'], fitted_files=full.get('fitted_files', []), startup_rows_removed=full['startup_rows_removed'], sampling_interval_seconds=full['sampling_interval_seconds'], source_projection_indices=selected, normalized_dtype='float32', command_smoothing='centered 51 rows, offline')

class MechanicalProjection:

    def __init__(self, array, indices):
        self.source = array
        self.indices = np.asarray(indices, np.int64)
        self.shape = (len(array), 6)
        self.dtype = np.dtype('float32')
        self.ndim = 2

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self[row][..., col]
        return np.ascontiguousarray(self.source[key][..., self.indices], dtype=np.float32)

def load_training_records(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / 'training_manifest.json').read_text())
    full = json.loads((dataset_dir / 'scaler.json').read_text())
    scaler = project_scaler(full)
    records = []
    for entry in manifest:
        path = resolve_array(entry, dataset_dir)
        if not path.is_absolute():
            path = dataset_dir / path
        array = np.load(path, mmap_mode='r')
        if array.shape != (entry['rows'], len(full['input_columns'])) or array.dtype != np.float32:
            raise ValueError(f'Invalid prepared array: {path}')
        train_rows = int(entry['train_rows'])
        validation_start = int(entry['validation_start'])
        if not 0 < train_rows == validation_start < len(array):
            raise ValueError(f'Invalid split: {path}')
        records.append(dict(array=MechanicalProjection(array, scaler['source_projection_indices']), rows=len(array), train_rows=train_rows, validation_start=validation_start))
    if not records:
        raise ValueError('No training recordings')
    return (records, scaler, manifest)

def load_test_rows(source, dataset_dir, scaler, out):
    full = json.loads((Path(dataset_dir) / 'scaler.json').read_text())
    projected = project_scaler(full)
    for key in ('input_columns', 'output_columns', 'mean', 'std', 'startup_rows_removed', 'sampling_interval_seconds'):
        if projected[key] != scaler[key]:
            raise ValueError(f'Prepared scaler differs from mechanical checkpoint: {key}')
    from common.evaluation import load_test
    array, cache = load_test(source, Path(dataset_dir), full, Path(out))
    return (MechanicalProjection(array, scaler['source_projection_indices']), cache)

@dataclass
class ModelConfig:
    latent_dim: int = 8
    hidden: int = 32

class MechanicalCompact(nn.Module):

    def __init__(self, config, scale):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(4, config.latent_dim)
        self.transition = nn.Linear(12 + config.latent_dim, config.latent_dim)
        self.hidden = nn.Linear(12 + config.latent_dim, config.hidden)
        self.decoder = nn.Linear(config.hidden, 4)
        self.skip = nn.Linear(12, 4, bias=False)
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        nn.init.normal_(self.decoder.weight, std=0.001)
        nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.skip.weight)

    def encode(self, current):
        return torch.tanh(self.encoder(current[:, STATE_INDICES]))

    def step(self, previous, current, latent):
        features = torch.cat((current, current - previous), -1)
        latent = torch.tanh(self.transition(torch.cat((features, latent), -1)))
        hidden = torch.relu(self.hidden(torch.cat((features, latent), -1)))
        state = current[:, STATE_INDICES] + (self.decoder(hidden) + self.skip(features)) * self.scale
        return (state, latent)

    def rollout(self, initial, future_commands):
        previous, current = (initial[:, 0], initial[:, 1])
        latent = self.encode(current)
        predictions = []
        for t in range(future_commands.shape[1]):
            state, latent = self.step(previous, current, latent)
            following = torch.empty_like(current)
            following[:, COMMAND_INDICES] = future_commands[:, t]
            following[:, STATE_INDICES] = state
            previous, current = (current, following)
            predictions.append(state)
        return torch.stack(predictions, 1)

def export_weights(model):
    return tuple((np.ascontiguousarray(p.detach().cpu().numpy()) for p in (model.encoder.weight, model.encoder.bias, model.transition.weight, model.transition.bias, model.hidden.weight, model.hidden.bias, model.decoder.weight, model.decoder.bias, model.skip.weight, model.scale)))

@njit(cache=True)
def native_encode(current, ew, eb):
    z = np.empty(len(eb), np.float32)
    for i in range(len(z)):
        value = eb[i]
        for j in range(4):
            value += ew[i, j] * current[STATE_ARRAY[j]]
        z[i] = np.tanh(value)
    return z

@njit(cache=True)
def native_step(previous, current, z, ew, eb, tw, tb, hw, hb, dw, db, skip, scale):
    features = np.empty(12, np.float32)
    for j in range(6):
        features[j] = current[j]
        features[j + 6] = current[j] - previous[j]
    next_z = np.empty(len(z), np.float32)
    for i in range(len(z)):
        value = tb[i]
        for j in range(12):
            value += tw[i, j] * features[j]
        for j in range(len(z)):
            value += tw[i, 12 + j] * z[j]
        next_z[i] = np.tanh(value)
    hidden = np.empty(len(hb), np.float32)
    for i in range(len(hidden)):
        value = hb[i]
        for j in range(12):
            value += hw[i, j] * features[j]
        for j in range(len(z)):
            value += hw[i, 12 + j] * next_z[j]
        hidden[i] = max(value, 0.0)
    state = np.empty(4, np.float32)
    for i in range(4):
        value = db[i]
        for j in range(len(hidden)):
            value += dw[i, j] * hidden[j]
        for j in range(12):
            value += skip[i, j] * features[j]
        state[i] = current[STATE_ARRAY[i]] + value * scale[i]
    return (state, next_z)

@njit(cache=True)
def native_chunk(rows, previous, current, z, recursive, mean, std, ew, eb, tw, tb, hw, hb, dw, db, skip, scale):
    previous = previous.copy()
    current = current.copy()
    z = z.copy()
    output = np.empty((len(rows), 4), np.float64)
    for k in range(len(rows)):
        if recursive:
            current[4] = rows[k, 4]
            current[5] = rows[k, 5]
        else:
            current = rows[k].copy()
        state, z = native_step(previous, current, z, ew, eb, tw, tb, hw, hb, dw, db, skip, scale)
        previous = current.copy()
        for j in range(4):
            current[STATE_ARRAY[j]] = state[j]
            output[k, j] = np.float64(state[j]) * std[STATE_ARRAY[j]] + mean[STATE_ARRAY[j]]
    return (output, previous, current, z)

@njit(cache=True)
def physical_step(previous, current, z, mean, std, ew, eb, tw, tb, hw, hb, dw, db, skip, scale):
    p = np.empty(6, np.float32)
    c = np.empty(6, np.float32)
    for j in range(6):
        p[j] = (previous[j] - mean[j]) / std[j]
        c[j] = (current[j] - mean[j]) / std[j]
    state, next_z = native_step(p, c, z, ew, eb, tw, tb, hw, hb, dw, db, skip, scale)
    result = np.empty(4, np.float64)
    for j in range(4):
        result[j] = np.float64(state[j]) * std[STATE_ARRAY[j]] + mean[STATE_ARRAY[j]]
    return (result, next_z)

def load_bundle(path=DEFAULT_ARTIFACT_DIR / 'best_model.pt'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('format') != FORMAT:
        raise ValueError('Expected mechanical-only Compact checkpoint')
    sc = ck['scaler']
    if sc['input_columns'] != INPUT_COLS or sc['output_columns'] != OUTPUT_COLS or len(sc['mean']) != 6 or (len(sc['std']) != 6):
        raise ValueError('Checkpoint is not a six-signal/four-state mechanical model')
    if not np.isfinite(sc['mean']).all() or not np.isfinite(sc['std']).all() or np.any(np.asarray(sc['std']) <= 0):
        raise ValueError('Invalid mechanical normalization')
    model = MechanicalCompact(ModelConfig(**ck['model_config']), ck['scale'])
    model.load_state_dict(ck['model'])
    model.eval()
    return (model, ck)

@njit(cache=True)
def held_command(previous, current, latent, command, steps, mean, std, *weights):
    previous = previous.copy()
    current = current.copy()
    latent = latent.copy()
    current[4] = (command[0] - mean[4]) / std[4]
    current[5] = (command[1] - mean[5]) / std[5]
    for _ in range(steps):
        state, latent = native_step(previous, current, latent, *weights)
        previous = current.copy()
        for j in range(4):
            current[j] = state[j]
    physical = np.empty(4, np.float64)
    for j in range(4):
        physical[j] = np.float64(current[j]) * std[j] + mean[j]
    return (physical, previous, current, latent)

class MechanicalPlant:

    def __init__(self, checkpoint=DEFAULT_ARTIFACT_DIR / 'best_model.pt'):
        self.model, self.checkpoint = load_bundle(checkpoint)
        self.weights = export_weights(self.model)
        sc = self.checkpoint['scaler']
        self.mean = np.asarray(sc['mean'], float)
        self.std = np.asarray(sc['std'], float)
        self.dt = float(sc['sampling_interval_seconds'])
        self.previous = None

    def reset(self, initial_physical):
        rows = np.asarray(initial_physical, float)
        if rows.shape != (2, 6) or not np.isfinite(rows).all():
            raise ValueError('reset requires two finite physical rows of six signals')
        normalized = ((rows - self.mean) / self.std).astype(np.float32)
        self.previous, self.current = (normalized[0].copy(), normalized[1].copy())
        self.latent = native_encode(self.current, self.weights[0], self.weights[1])
        return self.state()

    def state(self):
        if self.previous is None:
            raise RuntimeError('Reset plant first')
        return self.current[:4].astype(float) * self.std[:4] + self.mean[:4]

    def snapshot(self):
        if self.previous is None:
            raise RuntimeError('Reset plant first')
        return tuple((value.copy() for value in (self.previous, self.current, self.latent)))

    def restore(self, snapshot):
        previous, current, latent = snapshot
        if previous.shape != (6,) or current.shape != (6,) or latent.shape != (self.model.config.latent_dim,):
            raise ValueError('Snapshot dimensions differ')
        if any((value.dtype != np.float32 or not np.isfinite(value).all() for value in snapshot)):
            raise ValueError('Snapshot must contain finite float32 normalized state and memory')
        self.previous, self.current, self.latent = (value.copy() for value in snapshot)

    def advance(self, command, internal_steps=100):
        if self.previous is None:
            raise RuntimeError('Reset plant first')
        command = np.asarray(command, float)
        if command.shape != (2,) or not np.isfinite(command).all() or (not isinstance(internal_steps, (int, np.integer))) or (internal_steps < 1):
            raise ValueError('Two finite commands and positive integer step count required')
        state, p, c, z = held_command(self.previous, self.current, self.latent, command, internal_steps, self.mean, self.std, *self.weights)
        if not np.isfinite(state).all() or not np.isfinite(z).all():
            raise RuntimeError('Mechanical plant diverged')
        self.previous, self.current, self.latent = (p, c, z)
        return state

class Metrics:

    def __init__(self, std):
        self.std = np.asarray(std, float)
        self.n = 0
        self.mean = np.zeros(4)
        self.m2 = np.zeros(4)
        self.sse = np.zeros(4)
        self.sae = np.zeros(4)
        self.maximum = np.zeros(4)
        self.persistence = np.zeros(4)

    def add(self, actual, prediction, baseline):
        n = len(actual)
        mean = actual.mean(0)
        delta = mean - self.mean
        self.m2 += ((actual - mean) ** 2).sum(0) + delta ** 2 * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n
        error = prediction - actual
        self.sse += (error ** 2).sum(0)
        self.sae += abs(error).sum(0)
        self.maximum = np.maximum(self.maximum, abs(error).max(0))
        self.persistence += ((baseline - actual) ** 2).sum(0)

    def table(self):
        import pandas as pd
        variance = self.m2 / self.n
        ratio = np.full(4, np.nan)
        np.divide(self.sse / self.n, variance, out=ratio, where=variance > np.finfo(float).eps * np.maximum(1.0, self.mean ** 2 + variance))
        rmse = np.sqrt(self.sse / self.n)
        return pd.DataFrame(dict(feature=OUTPUT_COLS, RMSE=rmse, MAE=self.sae / self.n, R2=1 - ratio, max_abs_error=self.maximum, NRMSE_train_std=rmse / self.std, persistence_RMSE=np.sqrt(self.persistence / self.n)))
