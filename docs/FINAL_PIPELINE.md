# Final research pipeline

```mermaid
flowchart TD
 A[Raw crane recordings] --> B[Data preprocessing]
 B --> C[Full-state surrogate models]
 C --> D[Two-second recursive validation]
 D --> E[GRU selected for MPC]
 E --> F[Offline Cartesian MPC]
 F --> G[Independent Compact mechanical validation plant]
 G --> H[Training-only MPC teacher trajectories]
 H --> I[25,000 samples / 500 episodes]
 I --> J[MLP and GRU policy distillation]
 J --> K[Held-out internal closed-loop evaluation]
 K --> L[CPU computational-feasibility benchmark]
 L --> M[Final baseline GRU neural controller]
 K --> N[Weighted / delta-loss ablation]
 N --> O[Internal generalization worsened: keep baseline]
 O --> M
```

The mechanical plant also generates reachable references before MPC solves; optimizer candidates use only the full-state predictor. Plant feedback advances after the chosen action. Policy own hidden memory is distinct from observer/predictor memory. External Test supports the earlier dynamics/MPC recording experiment; all teacher/policy stages use Train source recordings only.

Evidence: [experiment map](EXPERIMENT_MAP.md), [final neural results](results/neural_controller_results.csv), [full-recording MPC results](results/mpc_final_results.csv), [ablation](results/improvement_study_results.csv).
