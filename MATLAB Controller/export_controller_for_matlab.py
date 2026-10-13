import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
from scipy.io import loadmat, savemat

PACKAGE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def numeric_reference(inputs, weights, mean, std):
    h = np.zeros(32, np.float32)
    result = []
    for physical in inputs:
        x = ((physical - mean) / std).astype(np.float32)
        gi = weights['gru_weight_ih'] @ x + weights['gru_bias_ih']
        gh = weights['gru_weight_hh'] @ h + weights['gru_bias_hh']
        r = np.float32(1) / (np.float32(1) + np.exp(-(gi[:32] + gh[:32])))
        z = np.float32(1) / (np.float32(1) + np.exp(-(gi[32:64] + gh[32:64])))
        n = np.tanh(gi[64:] + r * gh[64:])
        h = (h - n) * z + n
        head = np.maximum(weights['head_weight_1'] @ h + weights['head_bias_1'], np.float32(0))
        output = np.tanh(weights['head_weight_out'] @ head + weights['head_bias_out'])
        result.append(np.clip(output.astype(np.float64), -1, 1))
    return np.asarray(result)


def inspect_copy(folder):
    config = json.loads((folder / 'controller_config.json').read_text())
    for name, expected in config['package_sha256'].items():
        assert (folder / name).is_file() and sha(folder / name) == expected, name
    for name in config['required_runtime_files']:
        assert (folder / name).is_file(), name
    w = loadmat(folder / 'controller_weights.mat', simplify_cells=True)
    c = loadmat(folder / 'controller_config.mat', simplify_cells=True)
    assert sum(w[name].size for name in config['weight_shapes']) == 6898
    for name, shape in config['weight_shapes'].items():
        assert list(np.atleast_1d(w[name]).shape) == shape and np.isfinite(w[name]).all(), name
        assert w[name].dtype == np.float32, name
    assert c['input_mean'].shape == (32,) and c['input_std'].shape == (32,)
    assert np.all(c['input_std'] > 0)
    assert len(c['feature_order']) == 32 and len(c['state_order']) == 22
    for path in folder.glob('*.m'):
        source = path.read_text(encoding='utf-8')
        assert '../' not in source and '..\\' not in source
        assert not any(token in source for token in ['py.', 'pyenv(', 'Neural Controller/', 'deployment/'])
    return w, c


def main():
    parser = argparse.ArgumentParser(description='Export the frozen baseline GRU as plain MATLAB arrays and implementation-test references; no plant simulation')
    parser.add_argument('--repo-root', type=Path, default=PACKAGE.parent)
    parser.add_argument('--output-dir', type=Path, default=PACKAGE)
    parser.add_argument('--sequence-length', type=int, default=1000)
    parser.add_argument('--portable-temp-root', type=Path, help='Writable temporary-copy parent outside the repository; defaults to repository parent')
    args = parser.parse_args()
    if args.sequence_length < 100:
        parser.error('Use at least 100 sequential verification calls')
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'verification').mkdir(exist_ok=True)
    sys.path.insert(0, str(repo))
    from deployment.neural_controller import CraneNeuralController, STATE_NAMES, FEATURE_ORDER
    frozen_paths = [repo / 'Neural Controller/models/gru/policy.pt', repo / 'GRU Latent/artifacts/best_model.pt',
                    repo / 'MPC offline Model/artifacts/mechanical_plant/best_model.pt']
    before = {p.relative_to(repo).as_posix(): sha(p) for p in frozen_paths}
    controller = CraneNeuralController()
    policy = controller._policy
    memory = policy.model.memory
    assert memory.num_layers == 1 and not memory.bidirectional and memory.dropout == 0
    state = policy.model.state_dict()
    keys = {'gru_weight_ih': 'memory.weight_ih_l0', 'gru_weight_hh': 'memory.weight_hh_l0',
            'gru_bias_ih': 'memory.bias_ih_l0', 'gru_bias_hh': 'memory.bias_hh_l0',
            'head_weight_1': 'head.0.weight', 'head_bias_1': 'head.0.bias',
            'head_weight_out': 'head.2.weight', 'head_bias_out': 'head.2.bias'}
    assert set(state) == set(keys.values())
    weights = {name: state[key].detach().cpu().numpy().copy() for name, key in keys.items()}
    assert sum(value.size for value in weights.values()) == 6898
    schema_source = repo / 'Neural Controller/dataset/metadata/feature_schema.json'
    scaler_source = repo / 'Neural Controller/configs/input_scaler.json'
    shutil.copy2(schema_source, output / 'feature_schema.json')
    cfg = dict(input_mean=policy.mean.copy(), input_std=policy.std.copy(),
               command_lower_bounds=np.array([-1., -1.]), command_upper_bounds=np.array([1., 1.]),
               feature_order=np.asarray(FEATURE_ORDER, dtype=object).reshape(1, -1),
               state_order=np.asarray(STATE_NAMES, dtype=object).reshape(1, -1),
               reference_order=np.asarray(['x', 'z', 'vx', 'vz'], dtype=object).reshape(1, -1),
               input_size=32, state_size=22, hidden_size=32, head_size=16, output_size=2,
               parameter_count=6898, controller_interval_s=.1, reference_preview_s=np.array([.1, .2]),
               checkpoint_sha256=before['Neural Controller/models/gru/policy.pt'], scaler_sha256=sha(scaler_source),
               schema_sha256=sha(schema_source), seed=42, checkpoint_epoch=178, gate_order='reset,update,new',
               normalized_dtype='single', physical_dtype='double', command_units='V')
    savemat(output / 'controller_weights.mat', weights, do_compression=True, oned_as='column')
    savemat(output / 'controller_config.mat', cfg, do_compression=True, oned_as='column')
    required = ['CraneNeuralController.m', 'controller_instance.m', 'controller_step.m', 'reset_controller.m',
                'initialize_controller.m', 'controller_file_sha256.m', 'validate_controller_package.m',
                'controller_weights.mat', 'controller_config.mat', 'controller_config.json', 'feature_schema.json']
    if output != PACKAGE:
        for path in PACKAGE.glob('*.m'):
            shutil.copy2(path, output / path.name)
        if (PACKAGE / 'README.md').is_file():
            shutil.copy2(PACKAGE / 'README.md', output / 'README.md')
    metadata = {key: value for key, value in cfg.items() if key not in ['input_mean', 'input_std', 'feature_order', 'state_order', 'reference_order', 'reference_preview_s', 'command_lower_bounds', 'command_upper_bounds']}
    metadata.update(reference_preview_s=[.1, .2], command_lower_bounds=[-1., -1.], command_upper_bounds=[1., 1.],
                    source_policy='Frozen baseline GRU; exact exported tensors, no retraining',
                    startup='Explicit actual previously applied command required on first step or reset',
                    runtime_requires_python=False, runtime_requires_dataset=False,
                    required_runtime_files=required,
                    weight_shapes={name: list(value.shape) for name, value in weights.items()},
                    package_sha256={name: sha(output / name) for name in ['controller_weights.mat', 'controller_config.mat', 'feature_schema.json']})
    metadata.update(weights_mat_sha256=metadata['package_sha256']['controller_weights.mat'],
                    config_mat_sha256=metadata['package_sha256']['controller_config.mat'],
                    packaged_schema_sha256=metadata['package_sha256']['feature_schema.json'])
    save_json(output / 'controller_config.json', metadata)
    fixture = json.loads((repo / 'deployment/tests/equivalence_fixture.json').read_text())
    base = np.concatenate([np.asarray(ep['inputs'], dtype=np.float64) for ep in fixture['episodes']])
    inputs = np.tile(base, (int(np.ceil(args.sequence_length / len(base))), 1))[:args.sequence_length]
    controller.reset()
    expected = np.array([controller.step(x[:22], x[22:24], x[24:28], x[28:32]) for x in inputs])
    reference_inputs = dict(state22=inputs[:, :22], previous_command=inputs[:, 22:24],
                            reference_01=inputs[:, 24:28], reference_02=inputs[:, 28:32], physical_inputs=inputs,
                            sequence_length=len(inputs), hidden_reset_count=1,
                            sequence_description=f'Cyclic repetition of 24 existing internal Train-source fixture inputs; {len(inputs)} uninterrupted calls; numerical stress test, not a physical trajectory')
    savemat(output / 'verification/python_reference_inputs.mat', reference_inputs, do_compression=True)
    savemat(output / 'verification/python_reference_outputs.mat', dict(expected_commands=expected), do_compression=True)
    reconstructed = numeric_reference(inputs, weights, policy.mean, policy.std)
    difference = reconstructed - expected
    maximum = float(np.max(abs(difference)))
    assert maximum <= 1e-6, f'NumPy equation diagnostic differs from authoritative policy by {maximum}'
    portable = False
    temporary_parent = (args.portable_temp_root or repo.parent).resolve()
    temporary = temporary_parent / ('crane_matlab_portable_' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        assert temporary.resolve().is_relative_to(temporary_parent)
        copied = temporary / 'MATLAB Controller'
        shutil.copytree(output, copied)
        loaded_w, loaded_cfg = inspect_copy(copied)
        for name in weights:
            np.testing.assert_array_equal(loaded_w[name], weights[name])
        copied_inputs = loadmat(copied / 'verification/python_reference_inputs.mat', simplify_cells=True)['physical_inputs']
        copied_result = numeric_reference(copied_inputs, loaded_w, loaded_cfg['input_mean'], loaded_cfg['input_std'])
        np.testing.assert_allclose(copied_result, expected, atol=1e-6, rtol=0)
        copied_maximum = float(np.max(abs(copied_result - expected)))
        portable = True
    finally:
        assert temporary.resolve().is_relative_to(temporary_parent) and temporary.name.startswith('crane_matlab_portable_')
        shutil.rmtree(temporary)
    after = {p.relative_to(repo).as_posix(): sha(p) for p in frozen_paths}
    assert after == before, 'Frozen checkpoint changed'
    summary = dict(python_checkpoint_sha256=cfg['checkpoint_sha256'], parameter_count=6898,
                   sequence_length=len(inputs), matlab_executed=False, matlab_version=None,
                   max_abs_output_difference=None, mean_abs_output_difference=None, rmse_output_difference=None,
                   tolerance=1e-6, equivalence_pass=None, reset_repeatability_pass=None,
                   portable_folder_pass=portable, portable_folder_validation_scope='File completeness, hashes, MAT-array reload and static runtime dependency inspection only; MATLAB not executed',
                   portable_folder_matlab_pass=None, controller_interval_s=.1, high_fidelity_plant_tested=False,
                   frozen_checkpoint_hashes_before=before, frozen_checkpoint_hashes_after=after, frozen_checkpoint_hashes_unchanged=True,
                   numpy_equation_diagnostic=dict(max_abs_error=maximum, mean_abs_error=float(np.mean(abs(difference))),
                       RMSE=float(np.sqrt(np.mean(difference**2))), per_output_max_error=np.max(abs(difference), axis=0).tolist(),
                       passed=True, scope='Python/NumPy diagnostic only; not MATLAB verification'),
                   reloaded_MAT_numpy_diagnostic=dict(max_abs_error=copied_maximum, passed=True,
                       scope='Exported MAT arrays in column-major layout; NumPy execution, not MATLAB execution'),
                   matlab_availability='No executable found on PATH or in the standard Windows installation directory',
                   sequence_scope=reference_inputs['sequence_description'])
    save_json(output / 'verification/verification_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
