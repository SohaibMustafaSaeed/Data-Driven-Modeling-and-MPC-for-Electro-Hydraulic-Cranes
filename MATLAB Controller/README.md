# Standalone MATLAB GRU controller

Copy this entire folder next to the external crane model. The MATLAB runtime uses only files inside this folder and core MATLAB matrix operations. It requires no Python, PyTorch, original `.pt` checkpoint, dataset, MPC, observer, Compact plant, Java or Deep Learning Toolbox.

The policy is the frozen baseline GRU: input 32, hidden 32, ReLU head 16, two tanh outputs, 6,898 parameters, seed 42, checkpoint epoch 178. It controls the planar Cartesian crane-tip target x/z/vx/vz through lift/jib commands in V. Exported tensors retain their exact float32 values; normalization uses the exported train-only double-precision mean/effective std.

## Validation status

**MATLAB execution, Python-versus-MATLAB equivalence and MATLAB reset repeatability: NOT RUN.** No MATLAB executable was found on PATH or in the standard Windows installation directory. The package is prepared for MATLAB interface testing; high-fidelity Simulink plant validation has not been performed.

Exported arrays and file completeness were checked in a clean copy outside the repository. Reloaded MAT arrays were exercised in a Python/NumPy equation diagnostic, with maximum command difference **2.3841858e-7 V** versus the authoritative controller over 1,000 uninterrupted calls. The direct NumPy diagnostic maximum was **2.9802322e-7 V**. These are export diagnostics, not MATLAB results. Actual MATLAB differences remain null in [verification_summary.json](verification/verification_summary.json).

The MATLAB test uses a strict **1e-6 V** maximum-error threshold and fails if exceeded. It never automatically increases the tolerance. Platform/BLAS operation ordering can differ, so MATLAB execution is required before claiming numerical equivalence.

## Files

- [CraneNeuralController.m](CraneNeuralController.m): native stateful controller class.
- [initialize_controller.m](initialize_controller.m), [reset_controller.m](reset_controller.m), [controller_step.m](controller_step.m): one persistent controller per MATLAB session through `controller_instance.m`.
- `controller_weights.mat`: named single-precision GRU/head matrices and biases.
- `controller_config.mat`: double mean/std, bounds, feature/state/reference order and frozen metadata.
- [controller_config.json](controller_config.json): dimensions, SHA-256 checksums and package contract.
- [feature_schema.json](feature_schema.json): copied authoritative 32-feature schema, including units.
- [validate_controller_package.m](validate_controller_package.m): completeness, dimensions, finite values, scalar metadata, order and hashes.
- [controller_file_sha256.m](controller_file_sha256.m): core MATLAB SHA-256 implementation.
- [verify_python_equivalence.m](verify_python_equivalence.m): sequential command comparison, reset repeatability and input-validation tests.
- [example_controller_run.m](example_controller_run.m): one valid fixture call.
- [simulink_interface_example.m](simulink_interface_example.m): callback template; no plant is connected.
- [export_controller_for_matlab.py](export_controller_for_matlab.py): one-time repository-side export and Python reference generation.

## Startup and runtime

In MATLAB, from the directory containing this folder:

```matlab
addpath('MATLAB Controller');
validate_controller_package();
initialize_controller();
reset_controller(initialPreviousCommand);

% Once every 100 ms, with values supplied by the external plant/reference:
command = controller_step(state22, actualPreviousCommand, ref01, ref02);
u_lift = command(1);
u_jib = command(2);
```

Or use separate class instances for separate plant streams:

```matlab
ctrl = CraneNeuralController();
ctrl.reset(initialPreviousCommand);
command = ctrl.step(state22, actualPreviousCommand, ref01, ref02);
```

`state22` must be a real finite 22x1 or 1x22 vector in the schema's state order/units. `actualPreviousCommand` contains two values. `ref01` and `ref02` each contain [x z vx vz] at t+0.1 and t+0.2 s. Commands are returned as a 1x2 double row in physical V, clipped to [-1,1]. Boom angles and reference frame must match the original state conventions; convert simulator units/relative-angle conventions outside the controller.

The first call requires the actual previous applied command, supplied to `reset` or `step`. No [0,0] startup is assumed. After a successful call, passing `[]` for `previousCommand` uses the last returned command. Supplying the actual applied command is recommended, especially with actuator limits, delays or overrides. Hidden state is initially single-precision zeros and remains internal between calls. Reset only at a new simulation/episode. Invalid inputs are rejected before advancing state.

## Exact PyTorch equations

The implementation follows the [PyTorch GRU specification](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html) and its [CPU hidden update](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/RNN.cpp): stacked rows are reset, update, new. For input/recurrent affine vectors `gi` and `gh`:

```text
r = sigmoid(gi_r + gh_r)
z = sigmoid(gi_z + gh_z)
n = tanh(gi_n + r .* gh_n)
h_new = (h_old - n) .* z + n
head = ReLU(W1*h_new + b1)
command = clip(double(tanh(Wout*head + bout)), -1, 1)
```

The reset gate multiplies the recurrent new-gate affine result **including its bias**, not the hidden vector before recurrent multiplication. Normalize the 32 physical inputs in double as `(x-mean)./effective_std`, then cast to single. GRU/head matrices, biases, hidden state and activations are single. No pre-GRU layer or output rescaling is added.

## Run MATLAB verification

```matlab
addpath('MATLAB Controller');
report = verify_python_equivalence();
example_controller_run();
```

References contain 1,000 uninterrupted calls with a single initial hidden reset. They cycle 24 already saved internal Train-source deployment-fixture inputs. The discontinuous cycle stresses numerical recurrence; it is not a new 100-second physical trajectory, teacher corpus or tracking experiment. Python reference commands were generated by the unchanged authoritative deployment controller.

The test prints max/mean absolute error, RMSE, per-output maxima and reset repeatability; checks row/column inputs, NaN/Inf, complex values, dimensions and first-command initialization; saves actual MATLAB metrics/version to `verification/verification_summary.json`; and fails when error exceeds 1e-6 V or reset outputs differ. The current summary explicitly distinguishes unexecuted MATLAB results from Python diagnostics.

For a real portability check, copy only this folder to a new location, start a fresh MATLAB session there, add only this folder, and run both validation functions. The current copied-folder check verified files, hashes, array reload and runtime dependencies without executing MATLAB.

## Future Simulink connection

Simulink supplies all 22 current signals directly; no learned observer is inserted. An external generator supplies the desired Cartesian previews. Evaluate once per **0.1 s / 10 Hz**, hold the two commands for that interval, and allow the high-fidelity plant to integrate with its own smaller solver steps. Do not advance GRU memory at repeated minor solver evaluations.

The callback template accepts `readState()`, `readPreview(time)` and `applyAndAdvance(command,0.1)`. The last callback returns the command actually applied, which becomes the next previous-command input. All callbacks are external placeholders; no Simulink model is opened or simulated here.

The class/functions support interpreted MATLAB script/interface use. A MATLAB System block or other custom Simulink integration can wrap them later. Generated-code MATLAB Function block compatibility is not claimed or implemented. Direct high-fidelity states can differ from the observer estimates used for training; independently validate performance and stability before use.

## Re-exporting

Python is needed only to regenerate the package from the authoritative repository, not to run the exported MATLAB controller:

```powershell
python -B "MATLAB Controller/export_controller_for_matlab.py"
```

From a copied package, pass `--repo-root` pointing to the original repository. `--output-dir`, `--sequence-length` (minimum 100) and `--portable-temp-root` are available. Exporting reinitializes MATLAB verification fields to NOT RUN, so rerun MATLAB verification after any export. Frozen policy, full-state GRU dynamics and mechanical-plant hashes are checked before/after export; accepted research results remain unchanged.
