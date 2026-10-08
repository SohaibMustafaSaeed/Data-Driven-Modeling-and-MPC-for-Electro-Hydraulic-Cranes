# Running the models and offline MPC

This directory is the project root. Scripts resolve artifacts relative to their
own location and do not import code from the original project folders.
Run the commands below from this directory using Python 3.12.

## Layout

```text
Final Codes/
├── Data Preprocessing/preprocess_dataset.py
├── Transformers/
│   ├── transformer.py
│   ├── test.py
│   ├── artifacts/
│   └── results/
├── GRU Latent/
│   ├── gru2.py
│   ├── test.py
│   ├── artifacts/
│   └── results/
├── LSTM/
│   ├── lstm4.py
│   ├── test.py
│   ├── artifacts/
│   └── results/
├── Compact Recurrent/
│   ├── trainCompactLatent.py
│   ├── test.py
│   ├── artifacts/
│   └── results/
├── MPC offline Model/
│   ├── offline_mpc.py
│   ├── reference_replay.py     # Recorded-command reproduction helpers
│   ├── plant/                 # Mechanical model, trainer, and evaluator
│   ├── artifacts/mechanical_plant/
│   └── results/
├── common/                    # Shared data, evaluation, and path utilities
├── compare_models.py
├── requirements.txt
├── requirements-lock.txt
└── package_manifest.json
```

All four full-state models and the independent mechanical plant include completed
trained checkpoints. Transformer completed 60 epochs, and all four models have
full-record cached evaluations with matched reset-500 horizons.

## Installation

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Use this environment's Python for the commands below; activating the environment
is another option. requirements-lock.txt records the exact installed versions
used for the package checks, including the CPU PyTorch build. Popup plots need
a graphical desktop and Python's Tk support. --no-show saves plots without a GUI.

## Dataset layout and paths

By default, put the raw recordings in this layout:

```text
data/
├── Train/Simulation_Output_00.csv ...
└── Test/Simulation_Output_53.csv ...
```

Alternatively, set the directory containing Train and Test:

```powershell
$env:IMOTION_DATA_ROOT = "D:\YourDataset\CSV"
```

This is the one raw-data path to change. The two short path comments in
common/project.py identify these settings. Use --test-file with an absolute
path to select a particular CSV; relative filenames are looked up under Test.
Raw CSVs and prepared arrays are not bundled or committed.

Each CSV must contain these columns, with these exact names:

```text
ulift[V], speedref[rad/s], speedpmsm[rad/s], iqref[A], udref[V], uqref[V],
id[A], iq[A], psid[Wb], psiq[Wb], pumppressure[Pa], plinepistonlift[Pa],
plinerodlift[Pa], ppistonsidelift[Pa], prodsidelift[Pa], thetalift[rad],
ujib[V], plinepistonjib[Pa], plinerodjib[Pa], ppistonjib[Pa], prodjib[Pa],
thetajib[rad], thetaliftdot[rad/s], thetajibdot[rad/s]
```

The loader orders the selected columns explicitly; extra columns are allowed.
Filenames must be unique across Train and Test. Rows must be chronological,
finite, and uniformly sampled at 10 microseconds. Files are enumerated by sorted
filename and retain independent sequence boundaries; records are not joined.

## Preprocessing

```powershell
python "Data Preprocessing/preprocess_dataset.py"
```

Explicit directories are also supported:

```powershell
python "Data Preprocessing/preprocess_dataset.py" --train-dir "D:\YourDataset\CSV\Train" --test-dir "D:\YourDataset\CSV\Test" --output-dir "D:\YourPreparedDataset"
$env:IMOTION_PREPROCESSED_DIR = "D:\YourPreparedDataset"
```

Defaults remove the first 1,800,000 rows (18 seconds) from each recording.
Within each retained training recording, the first 90% is training and the last
10% is validation. Normalization statistics use training portions only. Test
recordings contribute to neither fitting nor training. Prepared recordings are
24-column float32 memory-mapped .npy files; scaler.json and split manifests
record feature order, boundaries, and sampling interval. Manifest array paths
are relative to the prepared dataset, so the prepared folder can be relocated.
Historical checkpoint manifests are resolved by filename against your selected
prepared training directory rather than depending on the original Windows path.

Only the two command columns use a centered 51-row, order-3 Savitzky-Golay
filter. This uses future command samples and is an offline preprocessing
pipeline. Live causal-control performance is not established by these results.
Change startup trimming with --skip-initial-rows only when retraining models for
that preprocessing; included checkpoints retain their original preprocessing.

To reuse this workspace's already prepared data without copying it:

```powershell
$env:IMOTION_DATA_ROOT = "C:\Data\Nitin Project\Dataset1\datafinal\Imotiondata\CSV"
$env:IMOTION_PREPROCESSED_DIR = "C:\Data\Nitin Project\Dataset1\datafinal\Imotiondata\CSV\MPC based Controller\Dataset preprocessed"
```

## Training

All full-state models form 48 features from the current 24 signals and their
recent differences. They predict 22 normalized state increments and reconstruct
the next state by adding the increments to the current state. Commands are inputs.
All four retain a direct linear residual skip; their temporal memory mechanisms
differ. The packaged training/inference implementations preserve the original
architectures and checkpoint formats.

To retrain Transformer while preserving the included checkpoint:

```powershell
python "Transformers/transformer.py" --epochs 60 --output-dir "Transformers/artifacts/retrained"
```

The other artifacts directories already contain trained models. To retrain:

```powershell
python "GRU Latent/gru.py" --epochs 60 --overwrite
python "LSTM/LSTM.py" --epochs 60 --overwrite
python "Compact Recurrent/trainCompactLatent.py" --epochs 60 --overwrite
```

Use --output-dir with a new folder to preserve the bundled artifacts. Trainers
refuse a nonempty destination unless --overwrite is provided. GRU/LSTM archive
their previous checkpoints when overwriting; use a new destination for guaranteed
preservation for every architecture. New weights invalidate prior result hashes.

GRU2/LSTM4/Transformer default to recursive horizons 100/500/1000, sampled
20%/60%/20% after warmup, with angle and angular-velocity loss weights of 4.
Their best checkpoint uses weighted 500-step validation. The original Compact
trainer has its original two-horizon curriculum and checkpoint-selection rule;
these historical differences are printed by the comparison and are not erased.
Use --help for dimensions, training horizons, loss weights, and optimizer options.

Artifacts include best_model.pt, latest_model.pt, best_numba_weights.npz,
training_history.json, and config.json where available. Checkpoints contain
architecture configuration, model weights, residual scale, normalization, and
training provenance. Training uses PyTorch; deployed CPU evaluation uses Numba.

## Testing each model

Every retained sample is evaluated by default, including when short reset horizons
repeat across the whole recording. Plots now use every sample in min/max display
bands, with interactive zoom for detail. See [whole-recording coverage](FULL_RECORDING.md).


```powershell
python "GRU Latent/test.py" --reinit-every 20
python "LSTM/test.py" --reinit-every 20
python "Compact Recurrent/test.py" --reinit-every 20
python "Transformers/test.py" --reinit-every 20
```

Default test recording: Simulation_Output_53.csv. Each command evaluates measured
one-step prediction and recursive prediction with the selected reset interval.
Omit --reinit-every for continuous recursive evaluation; --mode measured or
--mode recursive selects one mode. These testing commands perform evaluation;
the comparison command below reuses matching completed evaluations.

Results go to each model's results/<recording>/checkpoint_<SHA256 prefix>/<mode>/.
They include physical-unit predictions, per-feature MAE/RMSE/R2/maximum error,
NRMSE divided by training standard deviation, persistence error, CPU call timing,
and amortized compiled-loop throughput. Warm compilation is excluded from timing.
--max-rows explicitly limits an evaluation for a quick check; it is not a full
recording result. --checkpoint, --dataset-dir, and --output-dir override defaults.

## Compare all four models and open separate plots

```powershell
python compare_models.py --reinit-every 20
```

The comparison prints all per-feature metrics and saved timing summaries for
measured one-step, continuous recursive, and reset-500 evaluation. Matching saved
results are reused. Missing evaluations are generated once; --saved-only disables
that inference. Default popups are 88 separate figures for reset-20: one model
versus recorded truth for each of 22 states. Close every window to finish.

```powershell
python compare_models.py --reinit-every 20 --plot-mode recursive
python compare_models.py --reinit-every 20 --plot-mode all
python compare_models.py --reinit-every 20 --no-show
```

--plot-mode all opens 264 figures for all three modes. Figures are also saved
under results/comparison/. Display thinning does not change the saved full-row
metrics. Checkpoint hashes, feature order, scaler, timestep, target range, reset
horizon, test source, and training split metadata are checked. Different historical
training objectives and benchmark dates remain visible in the printed provenance.

Bundled results include reset-500 for all four models. Historical Compact
reset-20 results are retained separately. Saved evaluation metadata uses project-relative checkpoint paths and
raw:/prepared: dataset paths; the original weights and prediction arrays are unchanged.
Embedded checkpoint metadata retains the historical training paths for provenance.

## Offline MPC

Whole-recording MPC is the default when --duration is omitted. Use --full-recording
to request this explicitly. It advances and scores every internal sample, including
the final partial control interval. See [full CSV commands](FULL_RECORDING.md).


The default objective now tracks the Cartesian end effector using exact PATU
local link vectors and absolute boom angles. See [Cartesian MPC](CARTESIAN_MPC.md)
for geometry, commands, objective weights, ten popup plots, and Cartesian metrics.
Use --tracking-space joint to select the original joint objective.

The full-state GRU predicts candidate futures. A separately trained, mechanical-only
Compact model acts as the independent simulated plant. Its 12 features are four
mechanical states, two commands, and their differences. It predicts four mechanical
state increments and uses no hydraulic/electrical inputs. Mechanical feedback
corrects the predictor after each committed action. The plant is not used inside
optimizer candidate rollouts.

Show the packaged fixed MPC experiment immediately, without re-solving or needing
raw data:

```powershell
python "MPC offline Model/offline_mpc.py" --plot-results "MPC offline Model/results/verified_one_second_v2"
```

Run a new one-second experiment with a reference generated by the mechanical plant:

```powershell
python "MPC offline Model/offline_mpc.py" --duration 1 --reference-mode plant-generated
```

To assess tracking of the original recording, use --reference-mode recorded.
The script first checks recorded-command reproduction; a failing reproduction
gate stops that experiment. Inspect the saved reproduction report before choosing
--allow-reference-mismatch. Plant-generated-reference success demonstrates this
simulated setup, not tracking of the original recording or a physical crane.

Defaults: 100 internal 10-microsecond samples per held control action (1 ms),
100 control intervals per prediction horizon (100 ms), ten command move blocks,
and bounded SLSQP optimization. This is an offline optimizer, not a demonstrated
1-ms real-time controller. --horizon-ms, --moves, --maxiter, costs, command bounds,
--start-seconds, and --test-file are configurable. Results are saved under
MPC offline Model/results/<timestamp>/ with tracking, model mismatch, reproduction,
optimizer diagnostics, trajectories, command plans, and plots. Windows stay open
until closed. --no-show saves plots only.

The mechanical plant's trained artifacts are bundled. To retrain/test it:

```powershell
python "MPC offline Model/plant/trainMechanicalCompact.py" --epochs 60 --overwrite
python "MPC offline Model/plant/testMechanicalCompact.py" --mpc-horizons 100 500 2000
```

## GitHub and saved results

The original source project remains intact. package_manifest.json lists copied
checkpoints, hashes, and result folders. The historical full-state predictions
occupy approximately 19 GB; the large .npy files require Git LFS. This directory
includes .gitattributes for .pt/.npy/.npz and .gitignore for datasets, environments,
temporary files, and compiled caches.

From this directory, initialize Git LFS before adding the files:

```powershell
git init
git lfs install
git add .
```

Publishing needs enough LFS storage for the bundled results. To publish only code
and checkpoints, exclude the model results directories from your Git commit;
evaluators can recreate them from the dataset. No repository or upload is created
by the packaging step.
