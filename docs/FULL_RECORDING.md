# Whole-CSV evaluation with 20-step resets and MPC

The main model analysis is now predict 20 internal steps, reset with measured data,
and repeat over the entire retained CSV. Continuous 82-second prediction is archived
as an optional diagnostic, not the default comparison. Public labels are GRU and LSTM.


Full recording means every sample retained by the existing preprocessing. The
18-second startup trim stays unchanged. Two measured rows initialize the recurrent
state and recent differences; every subsequent retained row is a prediction target.

For Simulation_Output_53.csv:

| Coverage | Count/range |
| --- | --- |
| Raw rows | 10,000,001 |
| Startup rows removed | 1,800,000 |
| Retained rows | 8,200,001 |
| Measured initialization rows | 2 |
| Prediction targets | 8,199,999 |
| Predicted CSV times | 18.000020 to 100.000000 s |
| Internal sample interval | 10 microseconds |

## Horizon and recording length are separate

A 500-step horizon is 5 ms at this sampling rate. Resetting from measurements
every 500 predictions repeats 16,400 blocks over the entire retained recording.
It does not limit evaluation to the first 500 samples. The same distinction applies
to MPC: its prediction horizon may be 100 ms while the controller runs for all
82 seconds of retained data.

The four full-state models already have saved complete-record evaluations for:

- Measured-input one-step prediction across the whole CSV.
- Continuous recursive prediction across the whole CSV, with no measurement resets.
- Recursive prediction reset every 500 samples, repeated across the whole CSV.

These are different deployment questions. A reset-based trace over 82 seconds
is not an 82-second autonomous open-loop rollout. Both are shown and labeled.

## Comparison, using existing full-record results

From the original CSV workspace:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\compareModels.py" --test-file "Simulation_Output_53.csv" --reinit-every 20 --saved-only
```

The default plot is the repeated 20-step case across the entire CSV. For all three methods,
use `--plot-mode all` (264 separate model/state figures plus error-over-time
overviews). Add `--no-show` to save without opening windows.

From the portable project root:

```powershell
python compare_models.py --test-file "Simulation_Output_53.csv" --reinit-every 20 --saved-only
```

The comparison prints covered/expected prediction counts and the full CSV time
range for every model/method. `recording_coverage.csv` verifies coverage;
`error_over_time.csv` reports per-state MAE/RMSE/R2/max error and normalized RMSE
in one-second windows. Global metrics still score every prediction. Each plotted
mode also has an error-over-time overview of the four mechanical states.

## Testing any individual model over the whole CSV

From the portable project root, with the dataset paths configured:

```powershell
python "GRU Latent/test.py" --test-file "Simulation_Output_53.csv" --reinit-every 20
python "LSTM/test.py" --test-file "Simulation_Output_53.csv" --reinit-every 20
python "Transformers/test.py" --test-file "Simulation_Output_53.csv" --reinit-every 20
python "Compact Recurrent/test.py" --test-file "Simulation_Output_53.csv" --reinit-every 20
python "MPC offline Model/plant/testMechanicalCompact.py" --test-file "Simulation_Output_53.csv" --mpc-horizons 100 500 2000
```

Omit `--reinit-every` for continuous recursive evaluation. Do not pass `--max-rows`
for full recording. That option deliberately creates a prefix test; coverage files
now explicitly label it incomplete. It remains useful for implementation checks,
not for reporting long-run model performance. Model testing performs inference;
the comparison command reuses matching saved results.

## Plots use all samples

The old plotting code selected uniformly spaced points. That already spanned the
whole time range, but could miss brief peaks between displayed samples. New plots
use means and minimum/maximum bands computed from every sample in each display
bin. Each full-recording envelope therefore includes all 8,199,999 predictions.

Use the standard Matplotlib zoom/pan toolbar to inspect a time interval. The traces
reload the visible data from the saved arrays and recompute display bins. Once the
visible range contains fewer samples than the display budget, the plot shows each
sample directly. Home restores the full-CSV view. `--plot-points` controls display
bins; it never truncates evaluated rows or saved predictions. Plots and aggregate
statistics remain manageable without opening 88 lines of millions of vertices.

## Full-CSV Cartesian MPC

Whole-recording mode is now the default when `--duration` is omitted. This runs
from the first retained measured pair to the last retained sample. It preserves
the existing GRU optimizer predictor, separate mechanical Compact plant, feedback
correction, Cartesian objective, actuator bounds, and controller/prediction horizons.

From the original CSV workspace:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\mpc_surrogate_mechanical_plant.py" --full-recording --test-file "Simulation_Output_53.csv" --tracking-space cartesian --reference-mode plant-generated --output-dir "MPC based Controller\full state model\mpc_mechanical_plant_runs\full_csv_patu" --overwrite
```

From the portable project root:

```powershell
python "MPC offline Model/offline_mpc.py" --full-recording --test-file "Simulation_Output_53.csv" --tracking-space cartesian --reference-mode plant-generated --output-dir "MPC offline Model/results/full_csv_patu" --overwrite
```

`--overwrite` archives existing result folders. Full-recording mode does not select
a high-activity excerpt. `--duration`, `--start-seconds`, and `--segment-from` are
episode-selection options; use them explicitly for a shorter controlled experiment.

At the default 100 internal samples per held action, this recording requires
82,000 optimizer solves. The last interval advances only 99 internal samples;
the final real CSV sample is included rather than silently dropped. The control
interval remains 1 ms except for this final partial interval. Optimizer forecasts
also use the correct partial interval near EOF.

The recorded reference beyond EOF holds its final pose with zero target velocity.
The plant-generated reference uses the final held recorded command for extra
lookahead. Those lookahead targets are not additional scored CSV data, and no
future measured hydraulic/electrical state refreshes the predictor.

Complete-record MPC is computationally expensive: 82,000 offline optimizations
can take many hours. Progress and ETA are printed. The horizon and objective have
not been shortened just to make the experiment finish faster. A failed or
interrupted run retains incomplete coverage status and is not presented as a
completed full-recording result.

## Every internal sample is saved and scored

Full MPC saves:

- `micro_predictor_states.npy`: all 22 uncorrected predicted physical states at
  every 10-microsecond sample.
- `micro_plant_states.npy` and `micro_reference_states.npy`: all four mechanical
  states at every internal sample.
- `micro_plant_tip.npy`, `micro_reference_tip.npy`, `micro_predictor_tip.npy`:
  Cartesian position/velocity at every internal sample.
- `recording_coverage.json`: expected/written microstep counts, CSV range, final
  partial block, and completion status.
- Global joint and end-effector RMSE/MAE/R2/max error computed from all internal
  samples. Control-endpoint metrics are retained separately for comparison.
- Controller trajectories, plans, commands, costs, solver timing, and forecasts.
  Full 22-state forecast horizons are stored separately in
  `predictor_horizons_full_state.npy` rather than duplicating them inside the archive.

The dense full-CSV plots show both booms together, each boom separately, and
end-effector tracking over the entire recording. Existing trajectory, geometry,
command, and solver figures are also preserved. Close all windows to finish.
`--no-show` saves all plots without opening them.

Reopen a completed run without re-solving:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\mpc_surrogate_mechanical_plant.py" --plot-results "MPC based Controller\full state model\mpc_mechanical_plant_runs\full_csv_patu"
```

## Validation and current evidence

The complete-record model comparison was run from cached predictions for all
four models and all three methods: every case covers 8,199,999 predictions.
All-sample peak preservation, raw zoom detail, numerically stable window metrics,
and sample counting were tested. Full-MPC execution was verified on a short
recording with 253 predictions and an exact 53-sample final partial block; its
committed dynamics matched the prior implementation exactly. This is a code-path
check, not a completed 82-second MPC performance measurement. Historical one-second
MPC galleries and 0.2-second A/B tests remain labeled as episode experiments.
