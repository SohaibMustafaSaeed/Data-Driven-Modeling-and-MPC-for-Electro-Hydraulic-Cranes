# Controller-time validation

## Prediction validation over 20 actual controller intervals

The stronger dynamics test uses the **actual 100-ms interval of the published
controller**, read from its saved configuration: **20 × 100 ms = 2 seconds**, or
**200,000 internal model steps**. This is separate from 20 internal samples
(0.2 ms) and 200 internal samples (2 ms). High R² in those tiny-horizon tests
does not establish predictive accuracy over an MPC-relevant duration.

All four models now have a recomputed **two-second recursive forecast test across
the entire retained CSV**: 8,199,999 targets each, 41 measured-reset blocks, and
199,999 predictions in the final partial block. Each block starts from two
measured rows and reset memory. There are no measured-state refreshes inside a
forecast block. Every retained target contributes to the global error metrics.

| Model | Lift angle RMSE [rad] | Jib angle RMSE [rad] | Lift velocity RMSE [rad/s] | Jib velocity RMSE [rad/s] |
| --- | --- | --- | --- | --- |
| GRU | 0.00267003 | 0.00419828 | 0.00434339 | 0.00561563 |
| LSTM | 0.00304465 | 0.00475421 | 0.00398889 | 0.00561928 |
| Transformer | 0.00340935 | 0.00841001 | 0.00503473 | 0.00952408 |
| Compact Recurrent | 0.00210751 | 0.00235846 | 0.00321849 | 0.0037923 |
| Hold initial measured state | 0.018517 | 0.0234549 | 0.00611215 | 0.00979436 |

The persistence baseline freezes all states at the initial measured row of each
two-second block. Errors are reported in physical units and relative to that
baseline. Angle R² can remain high because boom angle varies slowly across the
recording; angular velocity and pressure errors reveal additional limitations.
These results establish conditional predictive skill on this held-out recording,
not perfect model quality or general performance on other operating conditions.

![Two-second forecasts across the entire CSV: lift and jib angles](docs/figures/controller20_full_csv_angles.png)

![Two-second forecasts across the entire CSV: lift and jib angular velocities](docs/figures/controller20_full_csv_velocities.png)

### Error and persistence at every controller prediction interval

The first figure scores every internal sample within each 100-ms lead interval,
aggregated across all forecast blocks throughout the CSV. The second scores exact
100-ms endpoints at leads 1 through 20. An incomplete final interval is included
in all-sample metrics but excluded from exact-endpoint metrics: **819 exact
endpoints per model** are available, with 40 at the full two-second lead and 41
at each earlier lead. No target beyond the CSV is invented.

![Controller-interval forecast errors versus holding the measured state](docs/figures/controller20_interval_errors.png)

![Exact controller-endpoint forecast errors versus persistence](docs/figures/controller20_endpoint_errors.png)

**Input assumption:** this test supplies the known recorded, preprocessed command
sequence at its original sample rate. It evaluates dynamics conditional on a
command preview, not the optimizer's ability to choose commands. Those commands
are not forced into 100-ms holds; forcing them into holds while scoring against
the original recording would also introduce command-approximation error. The
existing centered command smoothing remains offline preprocessing. This test
therefore does not establish deployment causality or closed-loop MPC tracking.

**Controller distinction:** the completed mechanical-plant MPC run above still
uses a two-interval, 200-ms planning horizon. This new 20-interval test does not
change that optimizer, its commands, its convergence rate or its measured tracking
results. The model-validation horizon and closed-loop controller experiment are
reported separately. The mechanical plant remains the only simulated MPC plant;
the four state-model forecasts are not four end-effector tracking experiments.

[All-state errors, R² and persistence](docs/results/controller20_all_state_metrics.csv) ·
[errors at all 20 intervals and exact endpoints](docs/results/controller20_metrics_by_interval.csv) ·
[model summary](docs/results/controller20_model_summary.csv) ·
[whole-CSV coverage](docs/results/controller20_recording_coverage.csv) ·
[duration, command assumption and checkpoint provenance](docs/results/controller20_protocol.json).

From the project root, with dataset paths configured:

```powershell
python evaluate_controller_horizon.py --controller-horizon 20 --saved-only
python compare_models.py --controller-config "MPC offline Model/results/full_csv_patu_100ms/config.json" --controller-horizon 20 --saved-only
```

The first command prints cached metrics and opens the complete-recording overviews
and persistence/lead plots. The second opens a separate full-recording figure for
each model/state. Omit `--saved-only` in the first command to compute missing
forecasts. Pass another completed controller's `--controller-config` to convert
its actual interval duration automatically. Add `--no-show` to save without GUI.
