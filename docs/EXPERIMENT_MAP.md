# Experiment map

Accepted numerical evidence is preserved. Regeneration produces new local measurements and never overwrites these tables. Full-recording MPC metrics and five-second neural-episode metrics have different scopes.

## Two-second surrogate comparison

Script: [evaluate_controller_horizon.py](../evaluate_controller_horizon.py)

Checkpoint: [GRU Latent/artifacts/best_model.pt](../GRU%20Latent/artifacts/best_model.pt)

Configuration: [docs/results/controller20_protocol.json](../docs/results/controller20_protocol.json)

Input: held-out Simulation_Output_53.csv; whole retained CSV.

Result: [docs/results/dynamics_recursive_results.csv](../docs/results/dynamics_recursive_results.csv)

Figure: [docs/figures/model_comparison_2s_angles.png](../docs/figures/model_comparison_2s_angles.png)

## GRU versus Compact MPC

Script: [reproduce_mpc.py](../reproduce_mpc.py)

Checkpoint: [GRU Latent/artifacts/best_model.pt](../GRU%20Latent/artifacts/best_model.pt)

Configuration: [MPC offline Model/configs/final_gru_mpc.json](../MPC%20offline%20Model/configs/final_gru_mpc.json)

Input: same full recording and plant-generated reference; Compact checkpoint is Compact Recurrent/artifacts/best_model.pt.

Result: [docs/results/mpc_predictor_comparison.csv](../docs/results/mpc_predictor_comparison.csv)

Figure: [docs/figures/mpc_predictor_comparison.png](../docs/figures/mpc_predictor_comparison.png)

## Iteration study

Script: [consolidate_mpc.py](../consolidate_mpc.py)

Checkpoint: [GRU Latent/artifacts/best_model.pt](../GRU%20Latent/artifacts/best_model.pt)

Configuration: [MPC offline Model/configs/final_gru_mpc.json](../MPC%20offline%20Model/configs/final_gru_mpc.json)

Input: same short segment; fixed maxiter comparison 5/10/20.

Result: [docs/results/slsqp_iteration_study.csv](../docs/results/slsqp_iteration_study.csv)

Figure: [docs/figures/slsqp_iteration_study.png](../docs/figures/slsqp_iteration_study.png)

## Reachable-reference MPC

Script: [reproduce_mpc.py](../reproduce_mpc.py)

Checkpoint: [GRU Latent/artifacts/best_model.pt](../GRU%20Latent/artifacts/best_model.pt)

Configuration: [docs/results/mpc_reachable_config.json](../docs/results/mpc_reachable_config.json)

Input: 82-second retained recording, plant-generated reachable reference.

Result: [docs/results/mpc_final_results.csv](../docs/results/mpc_final_results.csv)

Figure: [docs/figures/mpc_reachable_tip_path.png](../docs/figures/mpc_reachable_tip_path.png)

## Recorded-reference MPC

Script: [reproduce_mpc.py](../reproduce_mpc.py)

Checkpoint: [GRU Latent/artifacts/best_model.pt](../GRU%20Latent/artifacts/best_model.pt)

Configuration: [docs/results/mpc_recorded_config.json](../docs/results/mpc_recorded_config.json)

Input: same recording, recorded trajectory, explicit reference-mismatch override.

Result: [docs/results/mpc_final_results.csv](../docs/results/mpc_final_results.csv)

Figure: [docs/figures/mpc_recorded_tip_path.png](../docs/figures/mpc_recorded_tip_path.png)

## Replay comparison

Script: [MPC offline Model/reference_replay.py](../MPC%20offline%20Model/reference_replay.py)

Checkpoint: [MPC offline Model/artifacts/mechanical_plant/best_model.pt](../MPC%20offline%20Model/artifacts/mechanical_plant/best_model.pt)

Configuration: [MPC offline Model/geometry_patu.json](../MPC%20offline%20Model/geometry_patu.json)

Input: held recorded commands; matching control endpoints.

Result: [docs/results/recorded_replay_vs_mpc.csv](../docs/results/recorded_replay_vs_mpc.csv)

Figure: [docs/figures/recorded_replay_vs_mpc.png](../docs/figures/recorded_replay_vs_mpc.png)

## Teacher dataset

Script: [Neural Controller/scripts/generate_teacher_dataset.py](../Neural%20Controller/scripts/generate_teacher_dataset.py)

Checkpoint: [GRU Latent/artifacts/best_model.pt](../GRU%20Latent/artifacts/best_model.pt)

Configuration: [Neural Controller/dataset/metadata/frozen_teacher.json](../Neural%20Controller/dataset/metadata/frozen_teacher.json)

Input: 500 frozen Train windows; manifest defines regions; independent mechanical plant checkpoint above.

Result: [docs/results/teacher_dataset_summary.csv](../docs/results/teacher_dataset_summary.csv)

Figure: [docs/figures/teacher_cartesian_workspace_coverage.png](../docs/figures/teacher_cartesian_workspace_coverage.png)

## MLP policy

Script: [Neural Controller/scripts/train_policies.py](../Neural%20Controller/scripts/train_policies.py)

Checkpoint: [Neural Controller/models/mlp/policy.pt](../Neural%20Controller/models/mlp/policy.pt)

Configuration: [Neural Controller/configs/training.json](../Neural%20Controller/configs/training.json)

Input: mpc_teacher_v1 frozen source split; seed 123.

Result: [docs/results/neural_controller_results.csv](../docs/results/neural_controller_results.csv)

Figure: [docs/figures/supplementary/neural_training_validation_loss_mlp.png](../docs/figures/supplementary/neural_training_validation_loss_mlp.png)

## GRU policy

Script: [Neural Controller/scripts/train_policies.py](../Neural%20Controller/scripts/train_policies.py)

Checkpoint: [Neural Controller/models/gru/policy.pt](../Neural%20Controller/models/gru/policy.pt)

Configuration: [Neural Controller/configs/training.json](../Neural%20Controller/configs/training.json)

Input: same corpus/split; seed 42 epoch 178; evaluate_policies.py runs observer-corrected plant feedback.

Result: [docs/results/neural_controller_results.csv](../docs/results/neural_controller_results.csv)

Figure: [docs/figures/neural_closed_loop_tip_trajectory_xy_or_xz.png](../docs/figures/neural_closed_loop_tip_trajectory_xy_or_xz.png)

## Latency benchmark

Script: [Neural Controller/scripts/evaluate_policies.py](../Neural%20Controller/scripts/evaluate_policies.py)

Checkpoint: [Neural Controller/models/gru/policy.pt](../Neural%20Controller/models/gru/policy.pt)

Configuration: [Neural Controller/configs/runtime_versions.json](../Neural%20Controller/configs/runtime_versions.json)

Input: 500 warm-up and 10,000 measured calls, CPU two threads; policy and full loop separate.

Result: [docs/results/neural_latency_results.csv](../docs/results/neural_latency_results.csv)

Figure: [docs/figures/neural_accuracy_latency_comparison.png](../docs/figures/neural_accuracy_latency_comparison.png)

## Focused improvement study

Script: [Neural Controller/improvement_study/scripts/run_study.py](../Neural%20Controller/improvement_study/scripts/run_study.py)

Checkpoint: [Neural Controller/improvement_study/models/selected_improved_gru.pt](../Neural%20Controller/improvement_study/models/selected_improved_gru.pt)

Configuration: [Neural Controller/improvement_study/config/weighting.json](../Neural%20Controller/improvement_study/config/weighting.json)

Input: same frozen corpus/split; validation-selected delta .20 seed 123; all75 internal episodes.

Result: [docs/results/improvement_study_results.csv](../docs/results/improvement_study_results.csv)

Figure: [docs/figures/improvement_episode_rmse_distribution_baseline_vs_improved.png](../docs/figures/improvement_episode_rmse_distribution_baseline_vs_improved.png)

## Additional provenance

[Table export hashes](results/consolidation_provenance.json), [source split](../Neural%20Controller/data_splits/source_split.json), [policy scaler](../Neural%20Controller/configs/input_scaler.json), [feature schema](../Neural%20Controller/dataset/metadata/feature_schema.json), [baseline bootstrap](../Neural%20Controller/results/comparison/paired_source_bootstrap.json) and [ablation bootstrap](../Neural%20Controller/results/improvement_study/bootstrap_metrics.json) provide finer traceability. The archives contain historical metadata snapshots; generation-stage status flags are not current project status.

## Dynamics artifacts by model

| Model | Trainer | Checkpoint | Configuration |
| --- | --- | --- | --- |
| GRU | [gru.py](../GRU%20Latent/gru.py) | [best_model.pt](../GRU%20Latent/artifacts/best_model.pt) | [config.json](../GRU%20Latent/artifacts/config.json) |
| LSTM | [LSTM.py](../LSTM/LSTM.py) | [best_model.pt](../LSTM/artifacts/best_model.pt) | [config.json](../LSTM/artifacts/config.json) |
| Transformer | [transformer.py](../Transformers/transformer.py) | [best_model.pt](../Transformers/artifacts/best_model.pt) | [config.json](../Transformers/artifacts/config.json) |
| Compact Recurrent | [trainCompactLatent.py](../Compact%20Recurrent/trainCompactLatent.py) | [best_model.pt](../Compact%20Recurrent/artifacts/best_model.pt) | [config.json](../Compact%20Recurrent/artifacts/config.json) |

Their shared two-second protocol and published mechanical-state metrics are linked above. Predictor-comparison reproduction profiles are `gru_compare` and `compact_compare` in `reproduce_mpc.py`; iteration profiles are archived experiment recipes, while the final teacher remains maxiter=5.
