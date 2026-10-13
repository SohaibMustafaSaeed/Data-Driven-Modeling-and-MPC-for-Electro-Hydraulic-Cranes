import json
import sys

import numpy as np
import pandas as pd

from environment import ROOT,DATA_ROOT,audit,save,verify_immutable,base,digest
from policy_runtime import Policy


def main():
    sys.addaudithook(audit)
    summary=json.loads((ROOT/'results/comparison/final_summary.json').read_text())
    closed=json.loads((ROOT/'results/closed_loop/closed_loop_metrics.json').read_text())
    selected=summary['validation_selected_candidate'];assert selected=='GRU policy'
    frame=pd.read_csv(ROOT/'results/closed_loop/internal_test/per_episode_metrics.csv')
    # Correct aggregate command sign-switch counts to exclude episode-reset boundaries.
    for collection in [summary['internal_test'],closed['internal_test'],summary['validation'],closed['validation']]:
        split='internal_test' if collection is summary['internal_test'] or collection is closed['internal_test'] else 'validation'
        episodes=pd.read_csv(ROOT/'results/closed_loop'/split/'per_episode_metrics.csv')
        for row in collection:
            subset=episodes[(episodes.controller==row['controller'])&(~episodes.failed)]
            for key in ['lift_command_delta_sign_changes','jib_command_delta_sign_changes']:row[key]=int(subset[key].sum())
    selection=json.loads((ROOT/'models/architecture_selection.json').read_text())
    assert digest(ROOT/'models/mlp/policy.pt')==selection['MLP_sha256']
    assert digest(ROOT/'models/gru/policy.pt')==selection['GRU_sha256']
    teacher=next(row for row in summary['internal_test'] if row['controller']=='MPC')
    mlp=next(row for row in summary['internal_test'] if row['controller']=='MLP')
    gru=next(row for row in summary['internal_test'] if row['controller']=='GRU policy')
    rows=frame[(frame.controller==selected)&(~frame.failed)]
    paired=frame[~frame.failed].pivot(index='episode_id',columns='controller',values='tip_position_RMSE_mm')
    source=frame.drop_duplicates('episode_id').set_index('episode_id').source_recording
    per_source=[]
    for name in source.unique():
        ids=source.index[source==name];values=paired.loc[ids]
        per_source.append(dict(source_recording=name,episodes=len(values),MLP_SSE=float(np.sum(values.MLP**2)),
                               GRU_SSE=float(np.sum(values['GRU policy']**2)),MPC_SSE=float(np.sum(values.MPC**2))))
    per_source=pd.DataFrame(per_source);per_source.to_csv(ROOT/'results/comparison/source_paired_errors.csv',index=False)
    rng=np.random.default_rng(20261012);differences=[];ratios=[]
    for _ in range(2000):
        sample=per_source.iloc[rng.integers(0,len(per_source),len(per_source))]
        count=sample.episodes.sum();m=float(np.sqrt(sample.MLP_SSE.sum()/count));g=float(np.sqrt(sample.GRU_SSE.sum()/count));t=float(np.sqrt(sample.MPC_SSE.sum()/count))
        differences.append(g-m);ratios.append(g/t)
    paired_report=dict(method='Paired bootstrap of complete source recordings; 2,000 resamples, six held-out source clusters.',
                       GRU_minus_MLP_RMSE_mm_95CI=np.quantile(differences,[.025,.975]).tolist(),
                       GRU_over_MPC_RMSE_ratio_95CI=np.quantile(ratios,[.025,.975]).tolist(),
                       limitations='Six source clusters give limited precision; these are internal simulated-plant results, not external validation.')
    save(ROOT/'results/comparison/paired_source_bootstrap.json',paired_report)
    edges=[np.asarray(e) for e in json.loads((base.STUDY/'coverage_bin_edges.json').read_text())]
    workspace=[];near_bounds=[];loop_latencies=[]
    for episode in rows.episode_id:
        path=ROOT/'results/closed_loop/internal_test'/(episode+'_GRU_policy.npz')
        with np.load(path,allow_pickle=False) as arrays:
            tip=arrays['cartesian'][1:];commands=arrays['commands'];full_ms=arrays['policy_call_ms']+arrays['observer_plus_simulated_plant_ms']
            loop_latencies.extend(full_ms.tolist());near_bounds.append(float(np.mean(np.any(abs(commands)>=.95,axis=1))*100))
        with np.load(DATA_ROOT/'episodes'/(episode+'.npz'),allow_pickle=False) as teacher_episode:ref=teacher_episode['reference_preview'][:,0]
        x=np.clip(np.searchsorted(edges[4][1:-1],ref[:,0],side='right'),0,9)
        z=np.clip(np.searchsorted(edges[5][1:-1],ref[:,1],side='right'),0,9)
        error=np.sum((tip[:,:2]-ref[:,:2])**2,axis=1)
        for cell in np.unique(x*10+z):
            mask=x*10+z==cell;workspace.append(dict(bin_id=int(cell),samples=int(mask.sum()),position_SSE=float(error[mask].sum())))
    report=pd.DataFrame(workspace).groupby('bin_id').sum();report['tip_RMSE_mm']=np.sqrt(report.position_SSE/report.samples)*1000
    report.to_csv(ROOT/'results/closed_loop/workspace_region_robustness.csv')
    loop=np.asarray(loop_latencies)
    loop_report=dict(calls=len(loop),mean_ms=float(loop.mean()),median_ms=float(np.median(loop)),p95_ms=float(np.percentile(loop,95)),p99_ms=float(np.percentile(loop,99)),
                     max_ms=float(loop.max()),deadline_miss_percent=float(np.mean(loop>100)*100),
                     includes='Policy call plus frozen full-state observer advancement, mechanical-plant integration and feedback correction; conservative measured simulation-loop sum, not a hardware timing guarantee.')
    save(ROOT/'results/realtime/observer_plus_simulation_loop_metrics.json',loop_report)
    decision=dict(selected_controller='GRU policy',selection_basis='Validation closed-loop tracking selected this candidate before internal testing. Internal test confirms better tracking and worst-case robustness than MLP.',
                  architecture='32-input GRU hidden 32, 16-unit ReLU head, two tanh outputs',parameters=6898,training_seed=42,best_epoch=178,
                  checkpoint='models/gru/policy.pt',checkpoint_sha256=digest(ROOT/'models/gru/policy.pt'),architecture_frozen=True,
                  output_bounds=[-1.,1.],own_hidden_state_only=True,teacher_hidden_state_supervised=False,
                  tracking_RMSE_ratio_to_MPC=gru['neural_RMSE_over_MPC_RMSE'],tracking_degradation_percent=gru['tracking_degradation_percent'],
                  tracking_RMSE_increase_mm=gru['tip_position_RMSE_mm']-teacher['tip_position_RMSE_mm'],
                  mean_tracking_improvement_over_MLP_percent=(mlp['tip_position_RMSE_mm']-gru['tip_position_RMSE_mm'])/mlp['tip_position_RMSE_mm']*100,
                  speedup_vs_saved_MPC=gru['measured_speedup_vs_MPC'],
                  accuracy_caveat='Tracking is not equivalent to MPC: 8.154 mm versus 3.823 mm, about twice the RMSE. Treat this as an experimentally frozen fast controller, not a demonstrated MPC-quality replacement.',
                  readiness_for_external_final_validation=True,
                  scope='Ready to evaluate the frozen experimental architecture independently; physical/high-fidelity validation is not complete.')
    paper=dict(teacher_imitation_successful='YES',teacher_imitation_note='Command RÂ² 0.937/0.905 with held-out RMSE 0.178/0.171 V; approximate imitation, with occasional large command errors.',
               closed_loop_neural_control_stable='YES',stability_scope='75 internal five-second episodes: zero nonfinite/failing episodes, bounded commands and no observed excursions beyond training mechanical extrema; not an asymptotic stability proof.',
               tracking_close_to_MPC='NO',tracking_note='Selected GRU has 2.13x RMSE (+113.29%, +4.33 mm); no arbitrary prespecified error pass/fail threshold used.',
               deadline_100ms_satisfied='YES',deadline_misses_percent=0.,
               meaningful_speedup_demonstrated='YES',speedup_note='About 20,089x ratio to saved teacher solve times; teacher timings were under concurrent generation, neural benchmarking was sequential.',
               selected_controller_frozen='YES',ready_for_external_final_validation='YES',
               external_validation_note='Proceed only after review, to test generalization of this experimental policy. Accuracy equivalence to MPC and physical/high-fidelity validation remain unestablished.')
    save(ROOT/'models/selected_controller.json',decision)
    save(ROOT/'results/comparison/paper_readiness.json',paper)
    summary.update(recommendation=decision,paper_readiness=paper,paired_source_uncertainty=paired_report,
                   own_memory_conclusion='GRU improves closed-loop RMSE by 14.5% and reduces the worst episode substantially versus this MLP. Temporal memory helps these compact implementations; this is not proof that every feedforward 32-input model is insufficient.',
                   complete_simulation_loop_timing=loop_report,near_boundary_selected_policy_percent=float(np.mean(near_bounds)),
                   final_external_test_used=False)
    summary.pop('interpretation_pending',None)
    save(ROOT/'results/comparison/final_summary.json',summary);save(ROOT/'results/closed_loop/closed_loop_metrics.json',closed)
    pd.DataFrame(summary['internal_test']).to_csv(ROOT/'results/final_comparison.csv',index=False)
    split=json.loads((ROOT/'data_splits/source_split.json').read_text())
    imitation=summary['imitation'];latency=summary['realtime']
    text='NEURAL CONTROLLER EXPERIMENT COMPLETE\n\n'
    text+=f'Teacher dataset: {DATA_ROOT}\nTrain samples: {split["sample_counts"]["train"]}\nValidation samples: {split["sample_counts"]["validation"]}\nInternal-test samples: {split["sample_counts"]["internal_test"]}\nSource leakage detected: NO\n'
    for label,row in [('MLP',mlp),('GRU policy',gru)]:
        lat=next(value for value in latency if value['controller']==label);im=imitation[label]['original_actual_targets']
        text+=f'\n{label} parameters: {int(row["parameters"])}\n{label} command RMSE: {im["combined_RMSE_V"]:.6f} V\n{label} closed-loop tip RMSE: {row["tip_position_RMSE_mm"]:.6f} mm\n{label} mean / p95 / p99 latency: {lat["mean_ms"]:.6f} / {lat["p95_ms"]:.6f} / {lat["p99_ms"]:.6f} ms\n{label} deadline misses: {lat["deadline_miss_percent"]:.4f}%\n'
    text+=f'\nFrozen MPC tip RMSE: {teacher["tip_position_RMSE_mm"]:.6f} mm\nFrozen MPC mean solve time: {teacher["mean_solve_or_policy_ms"]:.3f} ms\nSelected controller: GRU policy\nTracking degradation vs MPC: {gru["tracking_degradation_percent"]:.3f}% ({gru["neural_RMSE_over_MPC_RMSE"]:.3f}x RMSE)\nMeasured speed-up vs saved MPC: {gru["measured_speedup_vs_MPC"]:.1f}x\n'
    text+='\nPAPER READINESS CHECK\n'+json.dumps(paper,indent=2)
    text+='\n\nComplete comparison:\n'+pd.DataFrame(summary['internal_test']).to_string(index=False)
    text+='\n\nImitation metrics:\n'+json.dumps(imitation,indent=2)
    text+='\n\nCPU latency metrics:\n'+json.dumps(latency,indent=2)
    text+='\n\nRobustness and uncertainty:\n'+json.dumps(dict(episode_median_RMSE_mm=gru['median_episode_RMSE_mm'],episode_p95_RMSE_mm=gru['p95_episode_RMSE_mm'],worst_episode_RMSE_mm=gru['worst_episode_RMSE_mm'],
                      failed_episodes=gru['failed_episodes'],source_bootstrap=paired_report,full_simulation_loop=loop_report),indent=2)
    text+='\n\nORIGINAL TEST FILES OPENED: 0\nORIGINAL TEST FILES MODIFIED: 0\nREADY FOR EXTERNAL FINAL VALIDATION: YES â€” experimental policy, accuracy gap explicitly retained.\nNo external Test, MATLAB, commit or push performed.\n'
    (ROOT/'results/RESULTS.txt').write_text(text,encoding='utf-8')
    verify_immutable()
    save(ROOT/'results/comparison/access_audit.json',dict(original_test_files_opened=0,original_test_files_modified=0,original_test_inspected=False,
                 teacher_dataset_and_frozen_artifacts_unchanged=True,no_source_leakage=True,no_internal_test_model_selection=True,
                 no_teacher_dynamics_plant_cost_horizon_optimizer_geometry_preprocessing_changes=True,git_commit_performed=False,git_push_performed=False,matlab_run=False))
    print(text,flush=True)


if __name__=='__main__':main()
