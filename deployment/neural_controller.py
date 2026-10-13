from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name('controller_config.json')
SCHEMA_PATH = ROOT / 'Neural Controller/dataset/metadata/feature_schema.json'
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
FEATURE_ORDER = tuple(item['name'] for item in SCHEMA['supervised_inputs'])
STATE_NAMES = tuple(name.split('[', 1)[0] for name in FEATURE_ORDER[:22])


def _runtime():
    name = '_crane_authoritative_policy_runtime'
    if name not in sys.modules:
        path = ROOT / 'Neural Controller/scripts/policy_runtime.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _vector(value, size, name):
    if value is None:
        raise ValueError(f'{name} is required')
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must contain real numeric values')
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must contain {size} real numeric values') from exc
    if result.shape != (size,):
        raise ValueError(f'{name} must have shape ({size},); received {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError(f'{name} contains NaN or Inf')
    return result.copy()


class CraneNeuralController:
    """Stateful CPU controller; inputs and returned commands use physical units."""

    def __init__(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        paths = {}
        for key, artifact in self.config['artifacts'].items():
            path = (ROOT / artifact['path']).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError(f'Artifact must be repository-relative: {key}')
            if not path.is_file():
                raise FileNotFoundError(f'Missing {key}: {path}; fetch the released artifacts with git lfs pull')
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
                raise ValueError(f'Frozen {key} hash mismatch: {path}')
            paths[key] = path
        if paths['checkpoint'] != (ROOT / 'Neural Controller/models/gru/policy.pt').resolve():
            raise ValueError('Deployment requires the final baseline GRU checkpoint')
        if paths['schema'] != SCHEMA_PATH.resolve():
            raise ValueError('Deployment requires the authoritative feature schema')
        torch.set_num_threads(self.config['torch_threads'])
        self._policy = _runtime().Policy(paths['checkpoint'])
        self._policy.model.to(device='cpu', dtype=torch.float32)
        bundle = self._policy.bundle
        expected = json.loads(paths['model_configuration'].read_text(encoding='utf-8'))
        scaler = json.loads(paths['normalization'].read_text(encoding='utf-8'))
        if bundle['architecture'] != 'gru' or expected['architecture'] != 'gru':
            raise ValueError('The frozen policy must be GRU')
        if (bundle['seed'], bundle['best_epoch']) != (42, 178):
            raise ValueError('The frozen baseline checkpoint must be seed 42, epoch 178')
        if sum(p.numel() for p in self._policy.model.parameters()) != 6898:
            raise ValueError('The frozen policy must have 6,898 parameters')
        if self._policy.model.memory.input_size != 32 or self._policy.model.memory.hidden_size != 32:
            raise ValueError('Unexpected GRU dimensions')
        if tuple(bundle['feature_order']) != FEATURE_ORDER:
            raise ValueError('Checkpoint feature order differs from the authoritative schema')
        for key in ['mean', 'effective_std']:
            if not np.array_equal(np.asarray(bundle['scaler'][key]), np.asarray(scaler[key])):
                raise ValueError(f'Checkpoint normalization differs from train-only scaler: {key}')
        self._lock = threading.RLock()
        self.reset()

    def reset(self, previous_command=None):
        """Start a new episode; optionally supply its actual initial applied command."""
        previous = None if previous_command is None else _vector(previous_command, 2, 'previous_command')
        with self._lock:
            self._policy.reset()
            self._previous_command = previous

    def step(self, state22, previous_command=None, reference_01=None, reference_02=None):
        if isinstance(state22, Mapping):
            missing = set(STATE_NAMES) - set(state22)
            extra = set(state22) - set(STATE_NAMES)
            if missing or extra:
                raise ValueError(f'State names mismatch; missing={sorted(missing)}, unexpected={sorted(map(str, extra))}')
            state22 = [state22[name] for name in STATE_NAMES]
        state = _vector(state22, 22, 'state22')
        reference = np.stack((_vector(reference_01, 4, 'reference_01'), _vector(reference_02, 4, 'reference_02')))
        with self._lock:
            if previous_command is None:
                if self._previous_command is None:
                    raise ValueError('First step requires the actual previous_command; supply it to step() or reset()')
                previous = self._previous_command.copy()
            else:
                previous = _vector(previous_command, 2, 'previous_command')
            hidden_before = self._policy.hidden
            try:
                command = self._policy.step_components(state, previous, reference)
            except Exception:
                self._policy.hidden = hidden_before
                raise
            self._previous_command = command.copy()
            return command.tolist()
