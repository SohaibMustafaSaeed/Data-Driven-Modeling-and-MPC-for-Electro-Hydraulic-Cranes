import numpy as np
import torch
from common.data import OUTPUT_COLS, STATE_INDICES
from common.project import read_report

def read_result(folder, model_type, mode):
    report = read_report(folder / 'evaluation.json')
    if report['mode'] != mode:
        raise ValueError(f'Wrong evaluation mode: {folder}')
    predictions = np.load(folder / 'predictions_physical.npy', mmap_mode='r')
    count = report['prediction_rows'] if model_type == 'lstm' else report['predictions']
    if predictions.shape != (count, 22):
        raise ValueError(f'Incomplete predictions: {folder}')
    scaler = torch.load(report['checkpoint'], map_location='cpu', weights_only=True)['scaler']
    if scaler['output_columns'] != list(OUTPUT_COLS):
        raise ValueError('Output feature order differs')
    cache = np.load(report['cache'], mmap_mode='r')
    if model_type == 'lstm2':
        start = int(report.get('tf_warmup', 0)) + 2
    else:
        start = report.get('initial_measured_rows', 1 if model_type == 'lstm' else 2)
    if len(cache) < start + count:
        raise ValueError('Truth cache is shorter than saved predictions')
    if model_type == 'lstm':
        dt = report['sampling_interval_seconds']
        offset = report['startup_excluded_seconds']
    else:
        dt = report['dt_s']
        offset = report['excluded_seconds']
    return dict(report=report, prediction=predictions, cache=cache, start=start, dt=dt, offset=offset, mean=np.asarray(scaler['mean'])[list(STATE_INDICES)], std=np.asarray(scaler['std'])[list(STATE_INDICES)])
