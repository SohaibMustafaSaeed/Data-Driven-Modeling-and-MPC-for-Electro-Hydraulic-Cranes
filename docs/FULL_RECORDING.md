# Whole-recording methodology

Every primary full-recording test retains the unchanged 18-second startup trim. Simulation_Output_53.csv has 8,200,001 retained rows. Two measured initialization rows leave 8,199,999 scored targets from CSV time 18.000020 to 100 s.

The 2-ms test predicts 200 internal steps before measured/memory reset and is a short diagnostic. The main surrogate test predicts 20 actual controller intervals: 20 x 100 ms = 2 s, or 200,000 internal steps. It uses 41 reset blocks; the last has 199,999 predictions. Every target is scored. Interval-endpoint metrics exclude the incomplete final interval, giving 819 exact endpoints. Known recorded preprocessed commands are provided as preview; future state measurements are not fed inside a block.

The controller experiment is different: 100-ms held commands, a 200-ms planning horizon, two command blocks, GRU prediction and an independent mechanical Compact plant. The plant advances continuously; only its current mechanical feedback corrects the GRU state estimate. It is not reset to future recorded states. Both final runs cover all 8,199,999 targets and 820 updates. The final action advances 9,999 internal samples. References beyond EOF are lookahead only, not scored recording targets.

Primary full-MPC tracking scores every internal sample. The existing short five-second iteration study and saved recorded-command replay score control endpoints. Replay-versus-MPC improvements therefore compare matching endpoints; full-sample MPC values remain separate.

All published time figures use the whole retained recording. Every sample contributes to mean/min/max envelopes. Static exported figures preserve that coverage; raw-sample interactive zoom requires regenerated local arrays.

See the main README for exact commands and the final scientific results. No models were retrained during consolidation.
