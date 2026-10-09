# Dataset, preprocessing and optional training

Run from the repository root with Python 3.12. The main README contains the tested installation and experiment commands.

## Dataset layout

```text
data/
  Train/Simulation_Output_00.csv ... Simulation_Output_42.csv
  Test/Simulation_Output_53.csv
  preprocessed/                 generated locally, ignored by Git
```

Set IMOTION_DATA_ROOT to the parent of Train/ and Test/. Set IMOTION_PREPROCESSED_DIR to the prepared output directory. The defaults are project/data and project/data/preprocessed. The shipped experiments used 43 training recordings, with each recording's first 90% retained samples used for training/scaler fitting and its suffix used for validation. Keep the original held-out recording in Test/, never Train/. Raw data is not included.

CSV headers must match exactly. Extra columns may exist; the reader selects these 24 signals in this order:

```text
ulift[V]
speedref[rad/s]
speedpmsm[rad/s]
iqref[A]
udref[V]
uqref[V]
id[A]
iq[A]
psid[Wb]
psiq[Wb]
pumppressure[Pa]
plinepistonlift[Pa]
plinerodlift[Pa]
ppistonsidelift[Pa]
prodsidelift[Pa]
thetalift[rad]
ujib[V]
plinepistonjib[Pa]
plinerodjib[Pa]
ppistonjib[Pa]
prodjib[Pa]
thetajib[rad]
thetaliftdot[rad/s]
thetajibdot[rad/s]
```

Commands occupy columns 0 and 16. The other 22 signals are state targets. The mechanical plant selects thetalift, thetajib, thetaliftdot, thetajibdot and the two commands, and appends recent differences to form 12 features.

## Prepare

```powershell
$env:IMOTION_DATA_ROOT = (Resolve-Path ".\data").Path
$env:IMOTION_PREPROCESSED_DIR = Join-Path $env:IMOTION_DATA_ROOT "preprocessed"
python check_setup.py --data
python "Data Preprocessing/preprocess_dataset.py" --skip-initial-rows 1800000 --train-fraction 0.9
```

Preprocessing removes the first 18 seconds at 10-us sampling and writes float32 memory-mapped arrays, scaler.json and training/testing manifests. Normalization is training-only. Existing valid caches are reused; different raw-file signatures or normalization trigger regeneration. The two command columns use centered Savitzky-Golay smoothing (51 rows, polynomial order 3), so these are offline prepared signals and deployment causality is not claimed.

When evaluating shipped checkpoints, test normalization is taken from the checkpoint. A newly fitted training scaler must not be substituted for a shipped checkpoint's scaler. Physical reconstruction is normalized * std + mean. Use the original data and split to reproduce numerical results.

## Optional training

The trained artifacts are shipped through Git LFS. Training is not needed to view results or repeat evaluation/MPC. To train new models, write to new output directories so shipped checkpoints and hashes remain intact:

```powershell
python "GRU Latent/gru.py" --output-dir "results/new_training/gru"
python "LSTM/LSTM.py" --output-dir "results/new_training/lstm"
python "Transformers/transformer.py" --output-dir "results/new_training/transformer"
python "Compact Recurrent/trainCompactLatent.py" --output-dir "results/new_training/compact"
python "MPC offline Model/plant/trainMechanicalCompact.py" --output-dir "results/new_training/mechanical"
```

Inspect --help and the shipped artifacts/config.json and training_history.json for the actual architecture/curriculum. The published full-state Compact curriculum differs from newer trainers. Fresh optimization, seeds and software may not reproduce the shipped weights exactly. Changing model checkpoints creates a new experiment, not the frozen controller's reproduction.

## Generated results

Model prediction arrays go to each model's results/<recording>/checkpoint_<hash>/<mode>/ tree. MPC experiments go to MPC offline Model/results/<profile>/. These are local, regenerable and ignored. Publication tables and figures live in docs/results and docs/figures. Use report_results.py and view_results.py to inspect bundled evidence without raw arrays.
