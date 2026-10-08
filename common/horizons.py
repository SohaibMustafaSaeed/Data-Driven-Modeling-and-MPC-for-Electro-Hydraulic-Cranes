import json
import math
from pathlib import Path


def controller_protocol(config_path, sample_time, intervals=20):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    duration = float(config['control_interval_seconds'])
    ratio = duration / sample_time
    if intervals < 1 or not math.isfinite(duration) or duration <= 0 or not math.isclose(ratio, round(ratio), rel_tol=1e-10):
        raise ValueError('Controller interval must be a positive integer multiple of the model sample time')
    if not math.isclose(float(config['dt_seconds']), sample_time, rel_tol=1e-10):
        raise ValueError('Controller and model sample times differ')
    block = int(round(ratio))
    return dict(controller_intervals=intervals, control_interval_seconds=duration,
                samples_per_control_interval=block, internal_horizon_steps=block * intervals,
                prediction_duration_seconds=duration * intervals,
                controller_config=str(Path(config_path).resolve()),
                command_policy='recorded prepared command preview at the original sample rate',
                interpretation='conditional dynamics validation over the controller time horizon; not a closed-loop MPC run')
