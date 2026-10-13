# Frozen GRU controller for an external high-fidelity plant

This interface packages the selected baseline GRU policy (6,898 parameters, seed 42, epoch 178). It controls the planar Cartesian crane-tip trajectory **x, z, vx, vz** through **u_lift and u_jib**, in the experiment's physical command units (V). The Python interface is verified. MATLAB/Simulink validation has not been performed.

## Python interface

Run from the repository root, or put that root on Python's module search path:

```python
from deployment.neural_controller import CraneNeuralController

controller = CraneNeuralController()
controller.reset(previous_command=actual_initial_command)
command = controller.step(
    state22=current_state,
    previous_command=actual_previous_command,
    reference_01=[x1, z1, vx1, vz1],  # desired values at t + 0.1 s
    reference_02=[x2, z2, vx2, vz2],  # desired values at t + 0.2 s
)
u_lift, u_jib = command
```

`current_state` is a one-dimensional 22-value array/list, or a dictionary containing exactly the canonical names exported by `deployment.neural_controller.STATE_NAMES`. Dictionary insertion order does not matter. Arrays must already follow that order; matrices and unnamed vectors of another length are rejected. Missing/extra names, wrong dimensions and NaN/Inf raise `ValueError` before advancing memory.

The single ordering authority is [feature_schema.json](../Neural%20Controller/dataset/metadata/feature_schema.json); the [neural README](../Neural%20Controller/README.md#physical-interface) lists all 32 features and their units. `STATE_NAMES` is derived from its first 22 entries. The remaining slots are previous lift/jib commands, then x/z/vx/vz previews at +0.1 and +0.2 s. Supply physical values in the schema's units, including absolute boom angles, with Cartesian references in the same world frame as the research pipeline. Convert any different simulator units or relative joint conventions outside this wrapper.

The wrapper automatically verifies/loads the baseline checkpoint, scaler, feature schema and model configuration using repository-relative paths in [controller_config.json](controller_config.json). It delegates normalization, GRU inference, memory updates, output conversion and [-1,1] clipping to the existing [authoritative runtime](../Neural%20Controller/scripts/policy_runtime.py). Inference uses CPU float32; physical input/output handling uses float64. Initialization selects two Torch CPU threads for the process.

Runtime requires the released policy files and Python dependencies, **not** Train/Test recordings, teacher episodes, MPC, a dynamics predictor, an observer or the Compact plant. The schema is metadata, not a dataset array.

## Startup, recurrent memory and previous command

Create one controller per plant stream. Call `reset()` at the start of each new simulation/episode; it clears the GRU hidden state and previous-command cache. Every successful step advances its own hidden state once. No hidden state crosses the external interface.

The accepted experiment initialized the previous command from the actual episode seed command. Therefore the first deployment call requires the actual initial/previous applied command, either through `reset(previous_command=...)` or `step(previous_command=...)`. There is no assumed zero startup. Use [0,0] only when that is what the external plant actually applied.

After the first successful call, `previous_command=None` uses the last returned command. During high-fidelity validation, explicitly supply the command actually applied in the previous interval, including actuator limits, delays or overrides. Reset does not load recordings or generate references. Failed input checks preserve memory and command history.

## Intended Simulink loop

```text
At t = k * 0.1 s:
  Simulink supplies all 22 current physical signals
  External reference generator supplies x/z/vx/vz at t+0.1 and t+0.2 s
  Python evaluates the frozen GRU once
  Simulink applies u_lift/u_jib and holds them for the next 100 ms
  The plant integrates with its own finer internal solver steps
  Repeat at the next control instant
```

The intended control rate is **10 Hz / 100 ms**. Call the policy once per control instant in chronological order, rather than at every plant integration step or repeated solver evaluation: each call advances recurrent memory. The wrapper does not schedule time or construct a desired trajectory. It consumes the two reference previews supplied by MATLAB/Simulink.

Simulink must expose all 22 signals directly. This wrapper does not insert the learned full-state observer used in the research evaluation. Direct high-fidelity states may have a different distribution from the observer estimates used during training; independent high-fidelity validation is still required. Existing learned-model accuracy and paper conclusions remain unchanged.

## Thin MATLAB bridge

[matlab_bridge.py](matlab_bridge.py) holds one persistent controller per Python interpreter:

- `initialize_controller()` loads once; repeated calls preserve memory.
- `reset_controller(previous_command=None)` starts a new episode.
- `controller_step(state22, previous_command, reference_01, reference_02)` returns a Python list of two physical commands.

The following is a **connection example, not validated MATLAB/Simulink integration**. Choose a Python executable supported by your MATLAB release with the repository dependencies installed. The `pyenv`, module import and Python-list conventions follow [MathWorks' Python setup](https://www.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html) and [Python-call documentation](https://www.mathworks.com/help/matlab/call-python-libraries.html).

```matlab
% Define pythonExecutable and repositoryRoot for your installation.
pyenv('Version', pythonExecutable, 'ExecutionMode', 'OutOfProcess');
py.sys.path.insert(int32(0), repositoryRoot);
bridge = py.importlib.import_module('deployment.matlab_bridge');
bridge.initialize_controller();
bridge.reset_controller(py.list(num2cell(initialAppliedCommand(:).')));

% Once per 100-ms control instant; vectors must use canonical units/order.
commandPy = bridge.controller_step( ...
    py.list(num2cell(state22(:).')), ...
    py.list(num2cell(actualPreviousCommand(:).')), ...
    py.list(num2cell(reference01(:).')), ...
    py.list(num2cell(reference02(:).')));
command = double(py.numpy.array(commandPy));
u_lift = command(1);
u_jib = command(2);
```

Keep the Python interpreter/module instance alive across calls. One bridge serves one plant stream; use separate `CraneNeuralController` instances for multiple simulations. Select a Simulink connection mechanism supported by your release and enforce the 0.1-s control sample time. Python driving MATLAB Engine is another later integration option. ONNX/MATLAB model import would be a separate future task; no export or policy redesign is included here.

## Verification and API example

```powershell
python -B deployment/tests/test_controller_interface.py
python -B deployment/example_runtime.py
```

Tests compare 24 sequential inputs from two already evaluated internal Train-source episodes against both saved baseline commands and the authoritative runtime. They also verify reset reproducibility, retained history, named inputs, previous-command caching/override, invalid-input handling, persistent bridge behavior and runtime independence with dataset accesses blocked. [The tiny fixture](tests/equivalence_fixture.json) is test/example material and is unnecessary for deployed inference.

[verification_result.json](tests/verification_result.json) records results and a 50-warm-up/200-call latency smoke check. This catches wrapper overhead; it is not a new scientific benchmark and does not replace the paper's existing timings. No training, teacher generation, MPC solve, Compact-plant rollout or MATLAB session is performed by these tests.
