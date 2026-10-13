# Focused GRU loss ablation

The baseline GRU remains selected. Same architecture, split and scaler; boundary/large-transition weights of 1–3; training-only transition p75; delta weights 0.05, 0.10 and 0.20; seeds 42, 123 and 2026. Two loss-selected checkpoints per run were screened with fixed validation episodes; all 75 validation episodes selected the final configuration. Internal test did not tune selection.

Validation selected delta 0.20, seed 123 (6.873895-mm tip RMSE). Internal-test tip RMSE became 8.988572 mm versus baseline 8.154487 mm: 10.23% degradation. Worst-episode error also increased. KEEP BASELINE GRU. Deployed plant rollouts are non-differentiable; no rollout fine-tuning was performed.

Accepted [results](../results/improvement_study/RESULTS.txt), [selection](models/selection.json), [histories/checkpoints](models/) and [figures](figures/) are preserved.

After regenerating the teacher corpus, run from the repository root:

```powershell
python "Neural Controller/improvement_study/scripts/run_study.py"
```

Fresh outputs go to ignored `results/neural_improvement/`; `NEURAL_IMPROVEMENT_OUTPUT_DIR` can select another local output directory. Published baseline and ablation evidence remain read-only. The original benchmark and fresh ablation timings have different run contexts. The candidate full simulation loop had one first-step 427.2802-ms outlier; policy-only deadline misses were zero.
