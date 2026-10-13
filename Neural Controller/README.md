# Neural controller

The final policy is the baseline GRU, 6,898 parameters, seed 42, epoch 178. The MLP is the comparison policy. The focused weighted/delta-loss candidate is retained only as an ablation.

## Physical interface

Authoritative ordering: [feature_schema.json](dataset/metadata/feature_schema.json). Current full-state inputs are observer estimates with mechanical-plant feedback; future measured states and teacher memory are excluded. Inputs use the shipped train-only scaler. Targets are first MPC moves; the strict nominal-bound training projection is documented in [target_projection.json](data_splits/target_projection.json).

| Index | Feature | Unit |
| --- | --- | --- |
| 0 | speedref[rad/s] | rad/s |
| 1 | speedpmsm[rad/s] | rad/s |
| 2 | iqref[A] | A |
| 3 | udref[V] | V |
| 4 | uqref[V] | V |
| 5 | id[A] | A |
| 6 | iq[A] | A |
| 7 | psid[Wb] | Wb |
| 8 | psiq[Wb] | Wb |
| 9 | pumppressure[Pa] | Pa |
| 10 | plinepistonlift[Pa] | Pa |
| 11 | plinerodlift[Pa] | Pa |
| 12 | ppistonsidelift[Pa] | Pa |
| 13 | prodsidelift[Pa] | Pa |
| 14 | thetalift[rad] | rad |
| 15 | plinepistonjib[Pa] | Pa |
| 16 | plinerodjib[Pa] | Pa |
| 17 | ppistonjib[Pa] | Pa |
| 18 | prodjib[Pa] | Pa |
| 19 | thetajib[rad] | rad |
| 20 | thetaliftdot[rad/s] | rad/s |
| 21 | thetajibdot[rad/s] | rad/s |
| 22 | previous_ulift[V] | V |
| 23 | previous_ujib[V] | V |
| 24 | reference_1_x[m] | m |
| 25 | reference_1_z[m] | m |
| 26 | reference_1_vx[m/s] | m/s |
| 27 | reference_1_vz[m/s] | m/s |
| 28 | reference_2_x[m] | m |
| 29 | reference_2_z[m] | m |
| 30 | reference_2_vx[m/s] | m/s |
| 31 | reference_2_vz[m/s] | m/s |

Targets: `target_ulift[V]`, `target_ujib[V]`.

## Frozen data and training

The dataset has 25,000 samples, 500 episodes, 43 Train recordings. Source groups contain 350/75/75 intact episodes (17,500/3,750/3,750 samples), with zero leakage. [Source split](data_splits/source_split.json), [membership](data_splits/episode_membership.csv), [training settings](configs/training.json) and [scaler](configs/input_scaler.json) are released. Original external Test is blocked by neural-stage audit hooks.

## Reproduction

Run commands from the repository root. Set `IMOTION_DATA_ROOT` and `IMOTION_PREPROCESSED_DIR` as needed. Generation reuses [the frozen 500-episode manifest](dataset/manifests/planned_episode_manifest.json); exploratory pilot/balancing data are unnecessary for regeneration. Large generated arrays remain ignored.

```powershell
python "Data Preprocessing/preprocess_dataset.py" --training-only
python "Neural Controller/scripts/generate_teacher_dataset.py" --plan-only
python "Neural Controller/scripts/generate_teacher_dataset.py" --workers 8
python "Neural Controller/scripts/run_experiment.py" --stage all
python report_results.py --section neural
python view_results.py --experiment neural
```

The baseline experiment writes `results/neural_reproduction/`, preserving released checkpoints/evidence. `--stage evaluate` evaluates copies of released policies in a new output folder. `--stage audit` performs the applied-target replay audit when explicitly requested; it is not part of repository cleanup. The standalone runtime is [policy_runtime.py](scripts/policy_runtime.py). The teacher [engine](scripts/teacher_engine.py) preserves the accepted episode-generation equations and optimizer settings.

## Accepted results

| Metric | MPC | MLP | GRU |
| --- | --- | --- | --- |
| Parameters | 4098 | 8418 | 6898 |
| Tip RMSE [mm] | 3.823172 | 9.534184 | 8.154487 |
| Tip MAE [mm] | 2.631914 | 6.273265 | 5.627090 |
| Maximum tip error [mm] | 34.262038 | 82.050947 | 47.670713 |
| Median episode RMSE [mm] | 2.514535 | 5.592219 | 5.121687 |
| P95 episode RMSE [mm] | 6.365890 | 17.204889 | 16.678902 |
| Worst episode RMSE [mm] | 17.714844 | 46.401105 | 28.451049 |
| Deadline miss [%] | 100.000000 | 0.000000 | 0.000000 |

| Policy | Mean [ms] | P95 [ms] | P99 [ms] | Maximum [ms] | Deadline misses [%] |
| --- | --- | --- | --- | --- | --- |
| MLP | 0.036379 | 0.040600 | 0.054603 | 0.341700 | 0.0 |
| GRU policy | 0.069659 | 0.078400 | 0.110605 | 0.547400 | 0.0 |

Policy-only timings are separate from observer/plant simulation timing. Saved teacher timing used parallel generation; neural timing used isolated calls. The selected GRU has 2.132911 times MPC tracking RMSE. This is computational feasibility in learned-model simulation, with a retained accuracy gap.

## Ablation

The same-architecture weighted/delta study selected delta=0.20, seed 123 on validation, but degraded held-out tip RMSE by 10.23%. [Results](results/improvement_study/RESULTS.txt), [candidate checkpoint](improvement_study/models/selected_improved_gru.pt) and [reproduction](improvement_study/scripts/run_study.py) are retained. Keep baseline GRU. Ablation regeneration writes `results/neural_improvement/`.
