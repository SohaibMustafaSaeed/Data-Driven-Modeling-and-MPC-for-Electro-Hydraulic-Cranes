# Full-recording model tests and offline MPC

Every analysis keeps the existing 18-second startup trim. For Simulation_Output_53.csv,
10,000,001 raw rows become 8,200,001 retained rows. Two measured rows initialize the
model; all remaining 8,199,999 samples, from 18.000020 to 100 s, are scored.

## State-model validation

The main comparison uses 200 internal samples per recursive block (2 ms), followed
by a measured-state and memory reset. This gives 41,000 blocks, with 199 predictions
in the last block. No future measured states enter the predictions inside a block.
All four full-state models retain their original trained weights.

From the portable project root, with data paths configured:

```powershell
python compare_models.py --reinit-every 200 --saved-only
python summarize_comparison.py --reinit-every 200
```

The first command prints the comparison and opens a separate full-recording figure
for each model/state. The second displays whole-recording mechanical overviews,
persistence baselines and error at every prediction lead. Add --no-show to save only.
Use --reinit-every 100 for an optional 1-ms test. Individual model test scripts now
default to 200. Use --reinit-every 0 explicitly for continuous recursive prediction.

From the original CSV workspace:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\full state model\compareModels.py" --reinit-every 200 --saved-only
```

Global metrics score every retained target. Per-second error tables and mean/min/max
plot envelopes also consume every sample. Interactive time plots reload visible
samples during zoom. The display budget changes rendering, not metric coverage.
Plots use 1-R? in scientific notation so near-one values are not printed as perfect.

## Actual closed-loop MPC experiment

The completed run uses GRU inside the optimizer and a separate 1,060-parameter
mechanical Compact model as the plant. The latter contains only four mechanical
states and two commands, plus their differences. No hydraulic/electrical inputs
are used in the mechanical plant.

```powershell
python "MPC offline Model/offline_mpc.py" --full-recording --tracking-space cartesian --reference-mode plant-generated --block-steps 10000 --horizon 2 --moves 2 --maxiter 5 --output-dir "MPC offline Model/results/full_csv_patu_100ms" --overwrite
```

This configuration uses a 100-ms held command and a 200-ms prediction horizon. It
performs 820 optimizer solves; the last action advances 9,999 internal samples.
These are controller settings, not the 200-sample (2-ms) surrogate-test horizon.
The script's older 1-ms control defaults remain available when those flags are omitted.
The tested configuration, solve times and convergence rate are reported in the README.
It is an offline experiment and has not demonstrated real-time control.

The plant advances continuously after each selected action. Its mechanical feedback
corrects only the corresponding fields of the GRU state estimate. It is never reset
to the future recorded crane states. Optimizer candidates do not advance the plant.

The mechanical plant fails recorded-command reproduction over the whole recording,
so the run explicitly uses a known reachable plant-generated reference: replay of
held recorded commands through the independent mechanical plant. Reference preview
is supplied to MPC; the recorded commands are not used as its prescribed solution.
The README's main trajectory plots show reference versus MPC-controlled plant,
not four surrogates' forward-kinematic reconstructions of a measured-reset test.

## Print and reopen the completed MPC results

```powershell
python "MPC offline Model/offline_mpc.py" --plot-results "MPC offline Model/results/full_csv_patu_100ms"
```

From the original CSV workspace:

```powershell
& ".\.venv_fast2\Scripts\python.exe" "MPC based Controller\Final Codes\MPC offline Model\offline_mpc.py" --plot-results "MPC based Controller\Final Codes\MPC offline Model\results\full_csv_patu_100ms"
```

This loads results without re-solving, prints metrics, and opens whole-recording
reference/plant time plots and x-z paths for both booms together, lift alone, jib
alone and the end effector. Windows remain open until closed. No single selected
prediction horizon is featured as a full-recording result.

The result folder contains every internal predicted full state, mechanical plant
state, reference state and tip position/velocity in micro_*.npy arrays. Global
tracking and mismatch metrics use all internal samples. Endpoint metrics are saved
separately. recording_coverage.json proves completion and includes the exact tail.

## Validation

The 200-step batched kernels were checked for bitwise equality with the original
per-block kernels, correct reset boundaries and a partial final block. Perturbing
future measured states inside a block leaves its forecasts unchanged. All four
original checkpoint hashes match. Lead-step metric counting and partial-block
statistics were checked independently. The completed MPC run scores all retained
samples, rather than extrapolating a short episode's metrics to the full recording.
