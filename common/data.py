import json
from pathlib import Path
import numpy as np
from common.project import PREPARED_DIR, resolve_array
# Use --dataset-dir or IMOTION_PREPROCESSED_DIR to change the prepared data path.
DEFAULT_DATASET_DIR = PREPARED_DIR
INPUT_COLS = ['ulift[V]', 'speedref[rad/s]', 'speedpmsm[rad/s]', 'iqref[A]', 'udref[V]', 'uqref[V]', 'id[A]', 'iq[A]', 'psid[Wb]', 'psiq[Wb]', 'pumppressure[Pa]', 'plinepistonlift[Pa]', 'plinerodlift[Pa]', 'ppistonsidelift[Pa]', 'prodsidelift[Pa]', 'thetalift[rad]', 'ujib[V]', 'plinepistonjib[Pa]', 'plinerodjib[Pa]', 'ppistonjib[Pa]', 'prodjib[Pa]', 'thetajib[rad]', 'thetaliftdot[rad/s]', 'thetajibdot[rad/s]']
COMMAND_INDICES = (0, 16)
STATE_INDICES = tuple((i for i in range(24) if i not in COMMAND_INDICES))
OUTPUT_COLS = [INPUT_COLS[i] for i in STATE_INDICES]

def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def load_training_records(dataset_dir):
    training_dir = dataset_dir / 'training'
    manifest_path = dataset_dir / 'training_manifest.json'
    scaler_path = dataset_dir / 'scaler.json'
    if not training_dir.is_dir() or not manifest_path.is_file() or (not scaler_path.is_file()):
        raise FileNotFoundError(f'Prepared training data not found under {dataset_dir}; run Data Preprocessing/preprocess_dataset.py first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scaler = json.loads(scaler_path.read_text(encoding='utf-8'))
    if scaler.get('input_columns') != INPUT_COLS or scaler.get('output_columns') != OUTPUT_COLS:
        raise ValueError('Prepared dataset feature order does not match this model')
    records = []
    for entry in manifest:
        array_path = resolve_array(entry, dataset_dir)
        if not array_path.is_absolute():
            array_path = dataset_dir / array_path
        array = np.load(array_path, mmap_mode='r')
        if array.ndim != 2 or array.shape != (entry['rows'], 24) or array.dtype != np.float32:
            raise ValueError(f'Invalid prepared array: {array_path}')
        train_rows = int(entry['train_rows'])
        validation_start = int(entry['validation_start'])
        if not 0 < train_rows == validation_start < len(array):
            raise ValueError(f'Invalid train/validation boundaries in {array_path}')
        records.append(dict(array=array, train_rows=train_rows, validation_start=validation_start, rows=len(array)))
    if not records:
        raise ValueError(f'No prepared training recordings in {manifest_path}')
    return (records, scaler, manifest)
