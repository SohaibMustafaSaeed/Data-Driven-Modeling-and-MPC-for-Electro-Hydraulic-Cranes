# Controller-horizon model validation

The main surrogate test uses 20 x 100-ms controller intervals: 2 s or 200,000 internal 10-us steps. It is not the 200-ms planning horizon of the final controller. Every forecast block starts from two measured rows and reset memory; no measured states refresh the rollout inside a block. All 8,199,999 retained targets are scored, including the final partial forecast block.

Known preprocessed recorded commands are provided as preview at the original sample rate. This is conditional dynamics evaluation, not a test of optimizer command selection. Commands are not approximated by 100-ms holds during this surrogate test; doing so against the original recording would introduce command-approximation error. Centered command smoothing remains an offline operation.

Persistence freezes the initial measured states. Global physical-unit RMSE/MAE, R2, normalized RMSE, maximum errors and ratios to persistence are available in docs/results/controller20_all_state_metrics.csv. Complete interval-endpoint metrics use 819 targets per model, with 40 at full two-second lead and 41 at earlier leads. All-sample interval statistics still include the final partial interval.

```powershell
python evaluate_controller_horizon.py --controller-horizon 20 --no-show
python compare_models.py --controller-config "MPC offline Model/configs/final_gru_mpc.json" --controller-horizon 20 --saved-only
python report_results.py --section models
python view_results.py --experiment models
```

The evaluation command computes missing prediction arrays from the supplied dataset; --saved-only requires previously generated local arrays. The report/view commands use bundled lightweight exports and need no dataset. The 200-step/2-ms test is a short diagnostic only. High R2 under frequent measured resets does not establish long-horizon quality.

The new full-state Compact model has stronger standalone mechanical forecasts on this recording, but GRU has better tracking in the controlled MPC comparison. All trained weights and normalization are unchanged.
