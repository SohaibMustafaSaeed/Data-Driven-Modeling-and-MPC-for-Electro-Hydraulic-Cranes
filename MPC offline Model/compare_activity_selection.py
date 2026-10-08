import argparse
import inspect
import json
from pathlib import Path
import sys

import offline_mpc as mpc


def main():
    parser = argparse.ArgumentParser(description='Controlled activity-selector experiment', add_help=False)
    parser.add_argument('--activity-rule', choices=['old', 'physical'], default='physical')
    args, remaining = parser.parse_known_args()
    source = inspect.getsource(mpc.activity_segment)
    if args.activity_rule == 'old':
        corrected = 'np.asarray(x[indices][:, MECH_INPUT[2:]], float) * std[MECH_INPUT[2:]] + mean[MECH_INPUT[2:]]'
        previous = 'np.asarray(x[indices][:, MECH_INPUT[2:]], float) + mean[MECH_INPUT[2:]] / std[MECH_INPUT[2:]]'
        if corrected not in source:
            raise RuntimeError('Selector changed; review the experiment before overriding it')
        source = source.replace(corrected, previous)
        exec(compile(source, '<old activity selector>', 'exec'), mpc.__dict__)
    sys.argv = [str(Path(mpc.__file__)), *remaining]
    mpc.main()
    if '--output-dir' not in remaining:
        return
    out = Path(remaining[remaining.index('--output-dir') + 1])
    (out / 'activity_selector_source.py').write_text(source, encoding='utf-8')
    path = out / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    config['activity_rule_under_test'] = args.activity_rule
    config['ab_test_note'] = 'Only activity selection differs; model weights, geometry and objective are unchanged.'
    if args.activity_rule == 'old':
        config['activity_selection'] = dict(mode='automatic' if config['arguments']['start_seconds'] is None else 'explicit_start', velocity_units='physical rad/s divided by training std', reconstruction='normalized + mean/std')
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
