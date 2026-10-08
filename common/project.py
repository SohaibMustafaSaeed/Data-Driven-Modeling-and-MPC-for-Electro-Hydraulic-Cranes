import os
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
# Set IMOTION_DATA_ROOT to the directory containing Train/ and Test/.
RAW_DATA_DIR = Path(os.environ.get('IMOTION_DATA_ROOT', ROOT / 'data')).expanduser().resolve()
# Set IMOTION_PREPROCESSED_DIR to reuse an existing prepared dataset.
PREPARED_DIR = Path(os.environ.get('IMOTION_PREPROCESSED_DIR', ROOT / 'data/preprocessed')).expanduser().resolve()
MODEL_FOLDERS = {'gru2': 'GRU Latent', 'lstm4': 'LSTM', 'transformer': 'Transformers', 'compact': 'Compact Recurrent'}
for folder in [*MODEL_FOLDERS.values(), 'Data Preprocessing', 'MPC offline Model/plant']:
    location = str(ROOT / folder)
    if location not in sys.path:
        sys.path.append(location)

def basename(value):
    return PureWindowsPath(str(value)).name

def same_recording(left, right):
    return basename(left) == basename(right)

def manifest_signature(manifest):
    return sorted((basename(entry['source_csv']), entry['rows'], entry['train_rows'], entry['validation_start'], entry.get('validation_rows', entry['rows'] - entry['validation_start'])) for entry in manifest)

def resolve_array(entry, dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    filename = basename(entry['file'])
    candidates = [dataset_dir / 'training' / filename, dataset_dir / filename]
    path = Path(entry['file'])
    if not path.is_absolute():
        candidates.insert(0, dataset_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Prepared recording missing: {filename}; run Data Preprocessing/preprocess_dataset.py for this dataset')

def saved_path(value):
    value = str(value)
    if value.startswith('prepared:'):
        return PREPARED_DIR / value.split(':', 1)[1]
    if value.startswith('raw:'):
        return RAW_DATA_DIR / value.split(':', 1)[1]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def read_report(path):
    import json
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('checkpoint', 'mechanical_checkpoint', 'source_csv', 'cache'):
        if key in report:
            report[key] = str(saved_path(report[key]).resolve())
    return report
