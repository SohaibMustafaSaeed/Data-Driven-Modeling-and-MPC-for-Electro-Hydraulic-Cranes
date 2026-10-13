import argparse
import hashlib
import json
from pathlib import Path

from common.project import ROOT, RAW_DATA_DIR, PREPARED_DIR, basename


def main():
    parser = argparse.ArgumentParser(description='Check dependencies, shipped model artifacts, and optional dataset layout without training or inference')
    parser.add_argument('--data', action='store_true', help='Also validate raw CSV folders, required training filenames and signal headers')
    parser.add_argument('--test-only', action='store_true', help='With --data, check only the held-out test CSV; shipped checkpoints need no training data for inference')
    args = parser.parse_args()
    import torch
    import numpy
    import pandas
    import scipy
    import numba
    import matplotlib
    import pyarrow
    print('Dependencies:', {module.__name__: module.__version__ for module in [torch, numpy, pandas, scipy, numba, matplotlib, pyarrow]})
    print('Project:', ROOT)
    print('Raw data:', RAW_DATA_DIR)
    print('Prepared arrays:', PREPARED_DIR)
    manifest = json.loads((ROOT / 'package_manifest.json').read_text(encoding='utf-8'))
    required_training = set()
    artifacts = [*manifest['models'].values(), manifest['mechanical_plant']]
    for artifact in artifacts:
        path = ROOT / artifact['checkpoint']
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint {path}; run git lfs pull')
        if path.stat().st_size < 200:
            raise ValueError(f'Checkpoint appears to be a Git LFS pointer: {path}; run git lfs pull')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash differs from shipped result provenance: {path}')
        ck = torch.load(path, map_location='cpu', weights_only=True)
        required_training.update(basename(entry['source_csv']) for entry in ck['training_config']['training_manifest'])
        print('Verified:', path.relative_to(ROOT), 'epoch', ck['epoch'])
    policies=ROOT/'Neural Controller/configs/release_checkpoint_manifest.json'
    if policies.exists():
        for name,artifact in json.loads(policies.read_text()).items():
            path=ROOT/artifact['checkpoint']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:raise ValueError(f'Policy checkpoint changed: {path}')
            ck=torch.load(path,map_location='cpu',weights_only=True)
            print('Verified policy:',name,'parameters',ck['parameter_count'])
        schema=json.loads((ROOT/'Neural Controller/dataset/metadata/feature_schema.json').read_text())
        assert len(schema['supervised_inputs'])==32 and len(schema['supervised_targets'])==2
    if args.data:
        from common.data import INPUT_COLS
        for folder in (['Test'] if args.test_only else ['Train', 'Test']):
            if not (RAW_DATA_DIR / folder).is_dir():
                raise FileNotFoundError(f'Expected {RAW_DATA_DIR / folder}')
        missing = [] if args.test_only else [name for name in sorted(required_training) if not (RAW_DATA_DIR / 'Train' / name).is_file()]
        if missing:
            raise FileNotFoundError('Original training recordings needed for exact reproduction: ' + ', '.join(missing))
        test = RAW_DATA_DIR / 'Test/Simulation_Output_53.csv'
        files = ([] if args.test_only else [RAW_DATA_DIR / 'Train' / name for name in sorted(required_training)]) + [test]
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            columns = pandas.read_csv(path, nrows=0).columns
            missing = [name for name in INPUT_COLS if name not in columns]
            if missing:
                raise ValueError(f'{path.name}: missing columns {missing}')
        print('Verified raw dataset filenames and 24 required signal headers. Row coverage and normalization are checked during preprocessing/evaluation.')
    print('Setup check passed. No model training, inference or MPC optimization performed.')


if __name__ == '__main__':
    main()
