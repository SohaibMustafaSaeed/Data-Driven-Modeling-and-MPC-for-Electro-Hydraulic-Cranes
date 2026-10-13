import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deployment.neural_controller import CraneNeuralController, STATE_NAMES, _runtime
from deployment import matlab_bridge

FIXTURE = json.loads(Path(__file__).with_name('equivalence_fixture.json').read_text())
REPORT = {'scope': 'Implementation equivalence and lightweight smoke checks only; no new scientific benchmark or plant experiment',
          'MATLAB_Simulink_validation_performed': False, 'external_Test_files_opened': 0}


class ControllerInterfaceTests(unittest.TestCase):
    def test_saved_and_authoritative_sequence_equivalence(self):
        maximum = 0.0
        for episode in FIXTURE['episodes']:
            controller = CraneNeuralController()
            reference = _runtime().Policy(ROOT / 'Neural Controller/models/gru/policy.pt')
            for x, saved in zip(episode['inputs'], episode['expected_commands']):
                actual = controller.step(x[:22], x[22:24], x[24:28], x[28:32])
                authoritative = reference.step_components(x[:22], x[22:24], np.asarray(x[24:]).reshape(2, 4))
                np.testing.assert_array_equal(actual, authoritative)
                np.testing.assert_allclose(actual, saved, atol=1e-7, rtol=0)
                maximum = max(maximum, float(np.max(abs(np.asarray(actual) - saved))))
        REPORT['equivalence'] = {'steps': 24, 'episodes': 2, 'saved_commands_max_absolute_error_V': maximum,
                                 'authoritative_runtime_bit_exact': True, 'saved_command_tolerance_V': 1e-7}

    def test_reset_and_retained_memory(self):
        controller = CraneNeuralController()
        sequence = FIXTURE['episodes'][0]['inputs']
        def run():
            return [controller.step(x[:22], x[22:24], x[24:28], x[28:32]) for x in sequence]
        first = run()
        hidden = controller._policy.hidden.clone()
        without_reset = run()
        self.assertFalse(np.array_equal(hidden.numpy(), controller._policy.hidden.numpy()))
        self.assertFalse(np.array_equal(first, without_reset))
        controller.reset()
        np.testing.assert_array_equal(first, run())
        REPORT['reset_and_sequential_memory_passed'] = True

    def test_named_input_and_previous_command_cache(self):
        explicit, remembered = CraneNeuralController(), CraneNeuralController()
        for i, x in enumerate(FIXTURE['episodes'][0]['inputs']):
            state = dict(reversed(list(zip(STATE_NAMES, x[:22]))))
            expected = explicit.step(x[:22], x[22:24], x[24:28], x[28:32])
            actual = remembered.step(state, x[22:24] if i == 0 else None, x[24:28], x[28:32])
            np.testing.assert_array_equal(expected, actual)
        x = FIXTURE['episodes'][0]['inputs'][0]
        remembered.reset(previous_command=x[22:24])
        explicit.reset()
        np.testing.assert_array_equal(explicit.step(x[:22], x[22:24], x[24:28], x[28:32]),
                                      remembered.step(x[:22], None, x[24:28], x[28:32]))
        next_x = FIXTURE['episodes'][0]['inputs'][1]
        actual_applied_override = [0.1, -0.2]
        np.testing.assert_array_equal(
            explicit.step(next_x[:22], actual_applied_override, next_x[24:28], next_x[28:32]),
            remembered.step(next_x[:22], actual_applied_override, next_x[24:28], next_x[28:32]))
        REPORT['named_input_previous_cache_and_override_passed'] = True

    def test_input_errors_do_not_advance_memory(self):
        controller = CraneNeuralController()
        x = np.asarray(FIXTURE['episodes'][0]['inputs'][0])
        with self.assertRaisesRegex(ValueError, 'actual previous_command'):
            controller.step(x[:22], None, x[24:28], x[28:32])
        controller.step(x[:22], x[22:24], x[24:28], x[28:32])
        hidden = controller._policy.hidden.clone()
        previous = controller._previous_command.copy()
        cases = [(x[:21], x[22:24], x[24:28], x[28:32]),
                 (x[:22].astype(complex) + 1j, x[22:24], x[24:28], x[28:32]),
                 (x[:22].reshape(22, 1), x[22:24], x[24:28], x[28:32]),
                 (x[:22], [0], x[24:28], x[28:32]),
                 (x[:22], x[22:24], [0, 0, 0], x[28:32]),
                 (x[:22], x[22:24], x[24:28], None),
                 ({name: 0 for name in STATE_NAMES[:-1]}, x[22:24], x[24:28], x[28:32]),
                 ({**dict(zip(STATE_NAMES, x[:22])), 'extra': 0}, x[22:24], x[24:28], x[28:32])]
        for slot in range(4):
            for invalid in [np.nan, np.inf, -np.inf]:
                vectors = [x[:22].copy(), x[22:24].copy(), x[24:28].copy(), x[28:32].copy()]
                vectors[slot][0] = invalid
                cases.append(tuple(vectors))
        for case in cases:
            with self.subTest(case=str(case)[:80]), self.assertRaises(ValueError):
                controller.step(*case)
            np.testing.assert_array_equal(hidden.numpy(), controller._policy.hidden.numpy())
            np.testing.assert_array_equal(previous, controller._previous_command)
        REPORT['invalid_input_checks_passed'] = len(cases) + 1

    def test_persistent_bridge(self):
        x = FIXTURE['episodes'][0]['inputs'][0]
        matlab_bridge.initialize_controller()
        matlab_bridge.reset_controller(x[22:24])
        first = matlab_bridge.controller_step(x[:22], None, x[24:28], x[28:32])
        identity = id(matlab_bridge._controller)
        hidden = matlab_bridge._controller._policy.hidden.clone()
        matlab_bridge.initialize_controller()
        self.assertEqual(id(matlab_bridge._controller), identity)
        np.testing.assert_array_equal(hidden.numpy(), matlab_bridge._controller._policy.hidden.numpy())
        matlab_bridge.reset_controller(x[22:24])
        np.testing.assert_array_equal(first, matlab_bridge.controller_step(x[:22], None, x[24:28], x[28:32]))
        REPORT['persistent_bridge_passed'] = True

    def test_runtime_independent_of_dataset_and_mpc(self):
        code = '''import sys,os
from pathlib import Path
root=Path.cwd()
blocked=[root/'data',root/'Neural Controller/dataset/generated',root.parent/'Train',root.parent/'Test',root.parent/'MPC Controller Data/Neural Controller Dataset']
def audit(event,args):
 if event not in ['open','os.scandir','os.listdir']:return
 if args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=Path(os.fsdecode(args[0])).resolve()
  if any(path.is_relative_to(p.resolve()) for p in blocked):raise RuntimeError('Dataset access: '+str(path))
sys.addaudithook(audit)
from deployment.neural_controller import CraneNeuralController
c=CraneNeuralController()
x=c._policy.mean
c.step(x[:22],x[22:24],x[24:28],x[28:32])
assert not any(name in sys.modules for name in ['offline_mpc','teacher_engine','environment','training_study','mechanical_compact'])
print('No dataset/observer/MPC/plant access or imports')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        REPORT['dataset_MPC_observer_plant_independence_passed'] = True

    def test_latency_smoke(self):
        controller = CraneNeuralController()
        x = FIXTURE['episodes'][0]['inputs'][0]
        for _ in range(50):
            controller.step(x[:22], x[22:24], x[24:28], x[28:32])
        times = []
        for _ in range(200):
            start = time.perf_counter_ns()
            controller.step(x[:22], x[22:24], x[24:28], x[28:32])
            times.append((time.perf_counter_ns() - start) / 1e6)
        self.assertLess(max(times), 100)
        REPORT['latency_smoke'] = {'warmup_calls': 50, 'calls': 200, 'mean_ms': float(np.mean(times)),
                                   'p95_ms': float(np.percentile(times, 95)), 'maximum_ms': max(times),
                                   'deadline_ms': 100, 'purpose': 'Deployment overhead smoke check; paper timing remains unchanged'}


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ControllerInterfaceTests))
    REPORT['all_tests_passed'] = result.wasSuccessful()
    REPORT['tests_run'] = result.testsRun
    REPORT['policy_checkpoint_sha256'] = hashlib.sha256((ROOT / 'Neural Controller/models/gru/policy.pt').read_bytes()).hexdigest()
    Path(__file__).with_name('verification_result.json').write_text(json.dumps(REPORT, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
