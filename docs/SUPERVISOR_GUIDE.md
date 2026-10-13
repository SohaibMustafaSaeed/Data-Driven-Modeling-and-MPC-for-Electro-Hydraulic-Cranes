# Supervisor guide

This chronology explains the completed experiments. Accepted tables and checkpoint hashes preserve the measured evidence. Commands regenerate local measurements; repository consolidation runs no scientific experiment.

## 1. Research problem

Track the crane tip accurately while computing commands within a 100-ms interval. Offline MPC provides a teacher for a compact policy.

[Code](../Neural%20Controller/scripts/evaluate_policies.py) · [Configuration](../Neural%20Controller/configs/runtime_versions.json) · [Results](../docs/results/neural_latency_results.csv) · [Figure](../docs/figures/neural_accuracy_latency_comparison.png)

## 2. Crane data

The crane recording contains 22 physical states and two commands at 10-microsecond sampling. External surrogate/MPC testing and neural internal source splitting serve different purposes.

[Code](../evaluate_controller_horizon.py) · [Configuration](../docs/results/controller20_protocol.json) · [Results](../docs/results/dynamics_recursive_results.csv) · [Figure](../docs/figures/model_comparison_2s_angles.png)

## 3. Data preparation

Remove the first 18 seconds, smooth commands with the accepted centered filter, standardize using training prefixes and preserve chronological samples.

[Code](../Data%20Preprocessing/preprocess_dataset.py) · [Configuration](../GRU%20Latent/artifacts/config.json) · [Results](../docs/results/dynamics_recursive_results.csv) · [Figure](../docs/figures/model_comparison_2s_angles.png)

## 4. Learned dynamics models

Compare residual full-state GRU, LSTM, causal Transformer and Compact Recurrent models using identical signals and evaluation protocol.

[Code](../evaluate_controller_horizon.py) · [Configuration](../docs/results/controller20_protocol.json) · [Results](../docs/results/dynamics_recursive_results.csv) · [Figure](../docs/figures/model_comparison_2s_angles.png)

## 5. Why short-horizon validation was misleading

Twenty internal samples cover only 0.2 ms. Frequent measured resets and slowly varying states make R² appear nearly perfect. Persistence and physical-duration evaluation expose the limitation.

[Code](../evaluate_controller_horizon.py) · [Configuration](../docs/results/controller20_protocol.json) · [Results](../docs/results/dynamics_recursive_results.csv) · [Figure](../docs/figures/model_comparison_2s_angles.png)

## 6. Two-second recursive validation

Twenty 100-ms prediction intervals give two seconds. Score the whole retained recording in recursive blocks and reset only at block starts.

[Code](../evaluate_controller_horizon.py) · [Configuration](../docs/results/controller20_protocol.json) · [Results](../docs/results/dynamics_recursive_results.csv) · [Figure](../docs/figures/model_comparison_2s_angles.png)

## 7. Compact model versus GRU

Full-state Compact has the lowest two-second mechanical prediction errors. This does not establish which predictor is best inside an optimizer.

[Code](../evaluate_controller_horizon.py) · [Configuration](results/controller20_protocol.json) · [Results](results/dynamics_recursive_results.csv) · [Figure](../docs/figures/model_comparison_2s_velocities.png)

## 8. Cartesian MPC

Use exact PATU local vectors and absolute boom angles to track x/z/vx/vz. Optimize two bounded moves over 200 ms; SLSQP maxiter stays five.

[Code](../reproduce_mpc.py) · [Configuration](../docs/results/mpc_reachable_config.json) · [Results](../docs/results/mpc_final_results.csv) · [Figure](../docs/figures/mpc_reachable_tip_path.png)

## 9. Separate learned mechanical plant

The 1,060-parameter Compact model predicts four mechanical states from 12 current/difference features. It advances after choosing an action and corrects mechanical observer fields.

[Code](../MPC%20offline%20Model/plant/mechanical_compact.py) · [Configuration](../MPC%20offline%20Model/artifacts/mechanical_plant/config.json) · [Results](../docs/results/recorded_replay_vs_mpc.csv) · [Figure](../docs/figures/recorded_replay_vs_mpc.png)

## 10. MPC model-selection experiment

With identical reference and controller settings, GRU gives 18.674-mm tip RMSE versus Compact 40.328 mm. Select GRU despite its weaker standalone recursive result.

[Code](../reproduce_mpc.py) · [Configuration](../MPC%20offline%20Model/configs/final_gru_mpc.json) · [Results](../docs/results/mpc_predictor_comparison.csv) · [Figure](../docs/figures/mpc_predictor_comparison.png)

## 11. Final MPC results

The reachable-reference and recorded-reference runs cover the entire retained 82-second CSV. At matching endpoints, recorded-command replay RMSE is 108.129 mm and MPC 26.086 mm: 75.88% improvement.

[Code](../reproduce_mpc.py) · [Configuration](../docs/results/mpc_recorded_config.json) · [Results](../docs/results/mpc_final_results.csv) · [Figure](../docs/figures/mpc_recorded_tip_path.png)

## 12. Why MPC cannot run in real time

Full-recording solves take about 1.1 seconds; saved teacher solves average about 1.4 seconds under parallel generation. Both exceed 100 ms. A solver termination status alone does not establish tracking quality or deadline feasibility.

[Code](../Neural%20Controller/scripts/evaluate_policies.py) · [Configuration](../Neural%20Controller/configs/runtime_versions.json) · [Results](../docs/results/neural_latency_results.csv) · [Figure](../docs/figures/neural_accuracy_latency_comparison.png)

## 13. MPC teacher dataset

Frozen training-only selection produced 500 five-second episodes and 25,000 samples across 43 recordings. The fixed manifest regenerates the accepted regions without exploratory resampling.

[Code](../Neural%20Controller/scripts/generate_teacher_dataset.py) · [Configuration](../Neural%20Controller/dataset/metadata/frozen_teacher.json) · [Results](../docs/results/teacher_dataset_summary.csv) · [Figure](../docs/figures/teacher_cartesian_workspace_coverage.png)

## 14. Neural-controller inputs and outputs

Use 22 current full-state estimates, two previous applied commands and two x/z/vx/vz previews. Output the first lift/jib move. Predictor memory is diagnostic only.

[Code](../Neural%20Controller/scripts/train_policies.py) · [Configuration](../Neural%20Controller/dataset/metadata/feature_schema.json) · [Results](../docs/results/neural_controller_results.csv) · [Figure](../docs/figures/supplementary/neural_training_validation_loss_mlp.png)

## 15. MLP and GRU training

Train on 17,500 samples; select with 3,750 validation samples; evaluate 3,750 internal-test samples. Entire recording groups stay separate. Train-only normalization and three deterministic seeds are preserved.

[Code](../Neural%20Controller/scripts/train_policies.py) · [Configuration](../Neural%20Controller/configs/training.json) · [Results](../docs/results/neural_controller_results.csv) · [Figure](../docs/figures/neural_closed_loop_tip_trajectory_xy_or_xz.png)

## 16. Closed-loop neural evaluation

Initialize each five-second episode once, propagate the frozen observer and independent mechanical plant recursively, and reuse teacher references. GRU tip RMSE is 8.154487 mm; MLP is 9.534184 mm; MPC is 3.823172 mm.

[Code](../Neural%20Controller/scripts/train_policies.py) · [Configuration](../Neural%20Controller/configs/training.json) · [Results](../docs/results/neural_controller_results.csv) · [Figure](../docs/figures/neural_closed_loop_tip_trajectory_xy_or_xz.png)

## 17. Real-time timing results

MLP is faster than GRU. Both meet the 100-ms policy deadline. Selected GRU mean/p99 policy latency is 0.069659/0.110605 ms; the full simulated loop averages 21.535185 ms.

[Code](../Neural%20Controller/scripts/evaluate_policies.py) · [Configuration](../Neural%20Controller/configs/runtime_versions.json) · [Results](../docs/results/neural_latency_results.csv) · [Figure](../docs/figures/neural_accuracy_latency_comparison.png)

## 18. Focused loss-improvement study

Validation selected weighted+delta .20, seed123, at6.873895 mm. Internal-test RMSE degraded to8.988572 mm and worst episode44.148859 mm. The validation gain did not transfer.

[Code](../Neural%20Controller/improvement_study/scripts/run_study.py) · [Configuration](../Neural%20Controller/improvement_study/config/weighting.json) · [Results](../docs/results/improvement_study_results.csv) · [Figure](../docs/figures/improvement_episode_rmse_distribution_baseline_vs_improved.png)

## 19. Final selected controller

Retain the baseline 6,898-parameter GRU, seed42 epoch178. It is a fast approximate MPC policy with a documented tracking gap, not MPC-equivalent accuracy.

[Code](../Neural%20Controller/scripts/train_policies.py) · [Configuration](../Neural%20Controller/configs/training.json) · [Results](../docs/results/neural_controller_results.csv) · [Figure](../docs/figures/neural_closed_loop_tip_trajectory_xy_or_xz.png)

## 20. Limitations

Evidence is learned-model simulation and computational feasibility. Hardware/Simulink testing, formal stability and external neural validation were not performed. Six internal source clusters limit uncertainty estimates; offline smoothing has a causality limitation.

[Code](../Neural%20Controller/improvement_study/scripts/run_study.py) · [Configuration](../Neural%20Controller/improvement_study/config/weighting.json) · [Results](../docs/results/improvement_study_results.csv) · [Figure](../docs/figures/improvement_episode_rmse_distribution_baseline_vs_improved.png)

See [EXPERIMENT_MAP.md](EXPERIMENT_MAP.md) for source/cohort scopes and exact artifact locations.
