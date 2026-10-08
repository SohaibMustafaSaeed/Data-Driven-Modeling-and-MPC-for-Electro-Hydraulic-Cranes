# Cartesian end-effector MPC with PATU geometry

The GRU still predicts crane dynamics and the independent mechanical Compact
model still supplies plant feedback. Forward kinematics maps their predicted
mechanical states to end-effector position and velocity. The optimizer now has
a Cartesian objective; this is not just a different plot of the joint objective.

## Geometry and frame

The supplied PATU assembly geometry is in `MPC offline Model/geometry_patu.json`:

| Quantity | Value |
| --- | --- |
| Base origin, x/z [m] | (-0.09, 1.4261) |
| Local lift-to-jib vector [m] | (2.886080943, 0.019892554) |
| Local jib-to-tip vector [m] | (2.035, 0.245) |
| Lift angle | Absolute relative to world horizontal |
| Jib angle | Absolute relative to world horizontal |
| Added angle offsets | Zero |

With the standard planar rotation matrix R, the elbow and crane tip are:

```text
p_elbow = base + R(theta_lift) @ lift_vector
p_tip   = p_elbow + R(theta_jib) @ jib_vector
```

The full local vectors include the approximately 0.395-degree lift and
6.865-degree jib geometric offsets. Adding those offsets again would count
them twice. The absolute recorded jib angle is used directly. Subtracting the
lift angle is only needed when assigning a relative joint coordinate in another
simulator such as MuJoCo; it is not part of this world-frame reconstruction.
Positive rotation is counterclockwise, x is horizontal, and z is vertical.
Tip velocities use the analytic Jacobian and the model's angular velocities.
This is a planar rigid-link model; slew and flexible deflection are not included.

## Run from the original CSV workspace

Both original and Final Codes entrypoints are updated. The original entrypoint
uses your existing dataset and trained artifacts without path configuration:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\mpc_surrogate_mechanical_plant.py" --tracking-space cartesian --reference-mode plant-generated --duration 1 --output-dir "MPC based Controller\full state model\mpc_mechanical_plant_runs\cartesian_patu" --overwrite
```

The exact geometry above loads automatically. This command creates a reachable
plant-generated reference and maps it through forward kinematics. `--overwrite`
archives an existing experiment folder before replacing it. It never replaces
the trained checkpoints. Solver execution remains offline and can take minutes
for a one-second episode; progress and solver timing are printed.

To print results and reopen all plots later without solving again:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\mpc_surrogate_mechanical_plant.py" --plot-results "MPC based Controller\full state model\mpc_mechanical_plant_runs\cartesian_patu"
```

## Run from the portable project root

Set IMOTION_DATA_ROOT and IMOTION_PREPROCESSED_DIR as described in RUNNING.md,
or preprocess the dataset under this project's data folder. Then:

```powershell
python "MPC offline Model/offline_mpc.py" --tracking-space cartesian --reference-mode plant-generated --duration 1 --output-dir "MPC offline Model/results/cartesian_patu" --overwrite
python "MPC offline Model/offline_mpc.py" --plot-results "MPC offline Model/results/cartesian_patu"
```

For the original recording's trajectory, choose `--reference-mode recorded`.
The existing recorded-command reproduction gate remains enabled for that case.
An independent learned plant cannot be assumed to reproduce every recording.

## Objective

Cartesian mode is now the default. It penalizes normalized tip position error,
tip velocity error, their terminal errors, command effort, and command changes.
The defaults are:

| Setting | Default |
| --- | --- |
| Position normalization | 0.01 m |
| Velocity normalization | 0.1 m/s |
| Position weight | 1 |
| Velocity weight | 0.1 |
| Terminal position weight | 5 |
| Terminal velocity weight | 1 |
| Joint angle/velocity and joint terminal weights | 0 in Cartesian mode |

Scales normalize errors; they are not hard tracking tolerances. Weights and
scales can be changed with `--position-scale`, `--cartesian-velocity-scale`,
`--position-weight`, `--cartesian-velocity-weight`, `--terminal-position-weight`,
and `--terminal-cartesian-velocity-weight`. Optional joint regularization remains
available with `--angle-weight` and `--velocity-weight`. To reproduce the previous
joint objective, use `--tracking-space joint` (original joint defaults are restored).

## Plots and metrics

All ten figure windows open together and remain open until closed:

- Both booms' joint states, selected forecast, commands/errors, and solver timing.
- Lift boom angle, angular velocity, and command in its own window.
- Jib angle, angular velocity, and command in its own window.
- Both boom-tip trajectories and the final two-link configuration together.
- Lift-tip Cartesian trajectory in its own window.
- Jib-tip/end-effector Cartesian trajectory in its own window.
- End-effector x/z and vx/vz tracking against time.

`--no-show` saves all figures without opening windows. Each run prints and saves
per-axis end-effector RMSE, MAE, R2, and maximum error in metres or m/s, plus
Euclidean position RMSE, mean distance error, maximum distance error, and final
distance error. The initial measured state is excluded from metrics. Constant
reference channels have undefined R2 rather than an invented perfect score.

Files include `end_effector_metrics.csv`, `end_effector_mismatch_metrics.csv`,
`boom_tip_metrics.csv`, `trajectory_cartesian.csv`, `kinematic_trajectories.npz`,
`kinematics_summary.json`, and `kinematics_results.txt`, alongside the original
joint-state metrics, full-state forecasts, solver history, and trajectories.
Model mismatch uses the uncorrected one-interval prediction, not feedback-corrected
states. Results preserve the exact geometry and objective for subsequent plotting.

## Optional externally specified tip path

`--cartesian-reference path.csv` accepts `time_s,x_m,z_m` and optional
`vx_m_s,vz_m_s`. Times are relative to the selected episode, must increase
strictly, and must cover the episode plus prediction lookahead. If velocities
are omitted, they are estimated from the interpolated position path.
Points outside the two-link geometric workspace are rejected. This does not
prove dynamic reachability or joint-limit feasibility; test feasible trajectories.

When an external path is used, the recorded/replayed joint curves are only
diagnostic guides. They are not inverse-kinematics targets, and the elbow path
is not uniquely specified by the end-effector path. Plot legends and saved
metadata identify that distinction. The external target bypasses the unrelated
recorded-reference reproduction gate; the reproduction diagnostic is still saved.

## Fixed model comparison

The previous CSV write failed because the Windows path was 264 characters long.
Default comparison directories now use a short combined checkpoint identifier,
with full model/checkpoint provenance saved inside. Existing evaluations are reused.

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\compareModels.py" --reinit-every 500 --saved-only
```

This prints all four models' metrics and opens 88 separate model/state plots.
Add `--plot-mode recursive` for continuous rollout plots or `--plot-mode all`
for 264 windows across all three modes. In the portable project use
`python compare_models.py --reinit-every 500 --saved-only`.

## Verification

Exact PATU vectors, absolute/relative angle options, analytic velocities against
finite differences, physical-unit Cartesian objective reconstruction, and reference
interpolation were checked. Short native-kernel MPC integrations ran through both
entrypoints. The full-record original comparison completed all three scenarios
from cached results without inference. Short integration accuracy is not a new
one-second MPC performance claim; run the command above to evaluate that episode.
