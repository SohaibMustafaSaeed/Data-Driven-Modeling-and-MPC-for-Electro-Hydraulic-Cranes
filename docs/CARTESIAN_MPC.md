# Cartesian MPC and PATU geometry

The final predictor is GRU (4,098 parameters). The independent simulated plant is the mechanical Compact Recurrent model (1,060 parameters). Candidate rollouts use only the full-state predictor; the plant advances after an action is selected. Mechanical feedback updates the last two mechanical-state rows and preserves other state estimates and predictor memory.

The final frozen settings are stored in MPC offline Model/configs/final_gru_mpc.json: dt 10 us, control interval 100 ms, horizon 200 ms, two move blocks, bounded SLSQP maxiter=5, fd-step=.03, ftol=1e-8. Position and velocity scales are .01 m and .1 m/s; Cartesian weights are 1,.1,5,1; effort .0001 and smoothness .001. Joint weights are zero in Cartesian mode. Exact sampled-training command bounds are retained in the config and passed explicitly by reproduce_mpc.py.

PATU base is (-.09,1.4261) m. Lift local vector is (2.886080943,.019892554) m; jib-to-tip vector is (2.035,.245) m. Both angles are absolute world-horizontal orientations. Full vectors already include geometric offsets. Analytic velocities use angular velocities and the FK Jacobian. No slew, flexibility or physical-hardware validation is represented.

The reachable experiment replays held recorded commands through the mechanical plant to generate a feasible target within that plant, then MPC tracks target preview without the generating future commands. The recorded experiment targets the original recorded mechanical/Cartesian trajectory and uses --allow-reference-mismatch to continue after the reproduction diagnostic fails. Replay failure is not a proof of unreachability.

The selected MPC is an offline teacher. Every measured solve exceeded its 100-ms interval. Formal SLSQP failure to converge is distinguished from action selection: the best finite bounded evaluated candidate is retained, including hold, warm start and finite-difference probes. Hold-cost improvement is a predicted objective statistic, not a plant counterfactual.

Optional external Cartesian references require time_s,x_m,z_m and optional vx_m_s,vz_m_s. Geometric workspace checks do not prove dynamic reachability. External targets are not part of the published frozen experiments.

For frozen reproduction, use reproduce_mpc.py. For bundled metrics/exports without the dataset, use report_results.py and view_results.py. For interactive local arrays after generating a run, use offline_mpc.py --plot-results.
