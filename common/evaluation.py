import json
from pathlib import Path
import numpy as np
import pandas as pd
from common.data import OUTPUT_COLS, COMMAND_INDICES

class Metrics:

    def __init__(self, std):
        self.std = std
        self.n = 0
        self.mean = np.zeros(22)
        self.m2 = np.zeros(22)
        self.sse = np.zeros(22)
        self.sae = np.zeros(22)
        self.maximum = np.zeros(22)
        self.persistence = np.zeros(22)

    def add(self, actual, prediction, baseline):
        n = len(actual)
        mean = actual.mean(axis=0)
        delta = mean - self.mean
        self.m2 += ((actual - mean) ** 2).sum(axis=0) + delta ** 2 * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n
        error = prediction - actual
        self.sse += (error ** 2).sum(axis=0)
        self.sae += abs(error).sum(axis=0)
        self.maximum = np.maximum(self.maximum, abs(error).max(axis=0))
        self.persistence += ((baseline - actual) ** 2).sum(axis=0)

    def table(self):
        var = self.m2 / self.n
        ratio = np.full(22, np.nan)
        np.divide(self.sse / self.n, var, out=ratio, where=var > np.finfo(float).eps * np.maximum(1.0, self.mean ** 2 + var))
        rmse = np.sqrt(self.sse / self.n)
        return pd.DataFrame(dict(feature=OUTPUT_COLS, MAE=self.sae / self.n, RMSE=rmse, R2=1 - ratio, max_abs_error=self.maximum, NRMSE_train_std=rmse / self.std, persistence_RMSE=np.sqrt(self.persistence / self.n)))

def load_test(source, dataset, scaler, out):
    from preprocess_dataset import convert_recording, read_chunks
    candidate = dataset / 'testing' / f'{source.stem}.npy'
    metadata = candidate.with_suffix('.json')
    skip = int(scaler['startup_rows_removed'])
    if candidate.exists() and metadata.exists():
        meta = json.loads(metadata.read_text(encoding='utf-8'))
        stat = source.stat()
        valid = Path(meta['source']).resolve() == source and meta['source_size'] == stat.st_size and (meta['source_mtime_ns'] == stat.st_mtime_ns) and (meta['skip_rows'] == skip) and (meta['mean'] == scaler['mean']) and (meta['std'] == scaler['std']) and (meta.get('smooth_window') == 51) and (meta.get('smooth_order') == 3) and (meta.get('smooth_columns') == list(COMMAND_INDICES))
        if valid:
            x = np.load(candidate, mmap_mode='r')
            if x.shape == (meta['rows'], 24) and x.dtype == np.float32:
                return (x, candidate)
    print(f'Preparing selected CSV only: {source}', flush=True)
    count = sum((len(c) for c in read_chunks(source, skip, 100000)))
    if count < 51:
        raise ValueError('Need at least 51 retained rows for the trained smoothing pipeline')
    path = convert_recording(source, out / 'data', count, skip, 100000, scaler)
    return (np.load(path, mmap_mode='r'), path)
