import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environment import ROOT,DATA_ROOT,FEATURE_SCHEMA,save,verify_immutable,load_episode,initial_rows,base,digest
from policy_runtime import Policy


def command_metrics(target,prediction):
    error=prediction-target;variance=np.var(target,axis=0)
    r2=[float(1-np.mean(error[:,j]**2)/variance[j]) if variance[j]>1e-12 else None for j in range(2)]
    return dict(samples=len(target),lift_MAE_V=float(np.mean(abs(error[:,0]))),lift_RMSE_V=float(np.sqrt(np.mean(error[:,0]**2))),
                jib_MAE_V=float(np.mean(abs(error[:,1]))),jib_RMSE_V=float(np.sqrt(np.mean(error[:,1]**2))),
                combined_RMSE_V=float(np.sqrt(np.mean(error**2))),R2_lift=r2[0],R2_jib=r2[1],maximum_absolute_error_V=float(abs(error).max()))


def tracking_metrics(reference,actual,reference_mech,actual_mech,commands,previous):
    error=actual-reference;distance=np.linalg.norm(error[:,:2],axis=1);delta=np.diff(np.vstack((previous,commands)),axis=0)
    mech=np.sqrt(np.mean((actual_mech-reference_mech)**2,axis=0))
    return dict(tip_position_RMSE_mm=float(np.sqrt(np.mean(distance**2))*1000),tip_position_MAE_mm=float(distance.mean()*1000),
                maximum_tip_error_mm=float(distance.max()*1000),final_tip_error_mm=float(distance[-1]*1000),
                x_RMSE_m=float(np.sqrt(np.mean(error[:,0]**2))),z_RMSE_m=float(np.sqrt(np.mean(error[:,1]**2))),
                vx_RMSE_m_s=float(np.sqrt(np.mean(error[:,2]**2))),vz_RMSE_m_s=float(np.sqrt(np.mean(error[:,3]**2))),
                Euclidean_tip_velocity_RMSE_m_s=float(np.sqrt(np.mean(np.sum(error[:,2:]**2,axis=1)))),
                lift_angle_RMSE_rad=float(mech[0]),jib_angle_RMSE_rad=float(mech[1]),lift_velocity_RMSE_rad_s=float(mech[2]),jib_velocity_RMSE_rad_s=float(mech[3]),
                lift_saturation_percent=float(np.mean(abs(commands[:,0])>=1-1e-6)*100),jib_saturation_percent=float(np.mean(abs(commands[:,1])>=1-1e-6)*100),
                mean_abs_delta_lift_V=float(np.mean(abs(delta[:,0]))),mean_abs_delta_jib_V=float(np.mean(abs(delta[:,1]))),
                maximum_command_change_V=float(abs(delta).max()),
                lift_command_delta_sign_changes=int(np.sum((delta[1:,0]*delta[:-1,0]<0)&(abs(delta[1:,0])>1e-6)&(abs(delta[:-1,0])>1e-6))),
                jib_command_delta_sign_changes=int(np.sum((delta[1:,1]*delta[:-1,1]<0)&(abs(delta[1:,1])>1e-6)&(abs(delta[:-1,1])>1e-6))))


def closed_episode(ctx,data,policy=None,replay=False):
    initial=initial_rows(data,ctx);ctx.plant.reset(initial[:,ctx.plant_mapping])
    previous=((initial[0]-ctx.mean)/ctx.std).astype(np.float32);current=((initial[1]-ctx.mean)/ctx.std).astype(np.float32)
    observer_memory=ctx.trainer.native_encode(current,ctx.weights[0],ctx.weights[1]);action_previous=data['previous_command'][0].copy()
    if policy:policy.reset()
    mechanical=[ctx.plant.state()];cartesian=[ctx.geometry.end_effector(mechanical[0])];actions=[];latency=[];transition=[];state_inputs=[]
    for k in range(50):
        full=current.astype(float)*ctx.std+ctx.mean
        start=time.perf_counter_ns()
        if replay:action=data['recorded_command'][k].copy()
        else:action=policy.step_components(full[ctx.core.SI],action_previous,data['reference_preview'][k])
        latency.append((time.perf_counter_ns()-start)/1e6)
        assert np.isfinite(action).all()
        state_inputs.append(np.concatenate((full[ctx.core.SI],action_previous,data['reference_preview'][k].reshape(8))))
        start=time.perf_counter_ns()
        if replay:
            next_mechanics=ctx.plant.advance(action,internal_steps=10000)
        else:
            previous,current,observer_memory,_,next_mechanics,_=ctx.core.advance_and_correct(
                ctx.rollout,previous,current,observer_memory,action,10000,ctx.mean,ctx.std,ctx.weights,ctx.plant)
        transition.append((time.perf_counter_ns()-start)/1e6)
        assert np.isfinite(next_mechanics).all()
        mechanical.append(next_mechanics);cartesian.append(ctx.geometry.end_effector(next_mechanics));actions.append(action);action_previous=action.copy()
    return dict(mechanical=np.asarray(mechanical),cartesian=np.asarray(cartesian),commands=np.asarray(actions),policy_call_ms=np.asarray(latency),
                observer_plus_simulated_plant_ms=np.asarray(transition),policy_inputs=np.asarray(state_inputs),time_s=np.arange(51)*.1)


def teacher_trace(data):
    return dict(mechanical=np.vstack((data['current_mechanical'][0],data['next_mechanical'])),
                cartesian=np.vstack((data['current_cartesian'][0],data['next_cartesian'])),commands=data['target_command'],
                policy_call_ms=data['solve_seconds']*1000,observer_plus_simulated_plant_ms=data['commit_seconds']*1000,time_s=np.arange(51)*.1)


def run_closed_split(ctx,split,policies):
    membership=pd.read_csv(ROOT/'data_splits/episode_membership.csv');episodes=membership[membership.split==split].episode_id.tolist()
    output=ROOT/'results/closed_loop'/split;output.mkdir(parents=True,exist_ok=True)
    rows=[];traces={name:[] for name in ['MPC','MLP','GRU policy','Recorded-command replay']};failures=[]
    envelope=json.loads((base.STUDY/'training_envelope_method.json').read_text())
    low=np.asarray(envelope['exact_min'][:4]);high=np.asarray(envelope['exact_max'][:4])
    started=time.perf_counter()
    for number,episode in enumerate(episodes):
        data=load_episode(episode);reference=data['reference_preview'][:,0];reference_mech=data['reference_next_mechanical']
        for label in traces:
            destination=output/(episode+'_'+label.replace(' ','_')+'.npz')
            try:
                if destination.exists():
                    with np.load(destination,allow_pickle=False) as saved:trace={key:saved[key] for key in saved.files}
                elif label=='MPC':trace=teacher_trace(data)
                else:trace=closed_episode(ctx,data,None if label=='Recorded-command replay' else policies[label],label=='Recorded-command replay')
                if not destination.exists():np.savez_compressed(destination,**trace)
                metrics=tracking_metrics(reference,trace['cartesian'][1:],reference_mech,trace['mechanical'][1:],trace['commands'],data['previous_command'][0])
                metrics.update(mean_policy_call_ms=float(trace['policy_call_ms'].mean()),
                               mean_observer_plus_simulated_plant_ms=float(trace['observer_plus_simulated_plant_ms'].mean()),
                               mechanical_samples_outside_training_envelope_percent=float(np.mean(np.any((trace['mechanical'][1:]<low)|(trace['mechanical'][1:]>high),axis=1))*100),
                               nonfinite_states=False,failed=False)
                rows.append(dict(split=split,controller=label,episode_id=episode,source_recording=str(data['source_recording'][0]),**metrics))
                traces[label].append((data,trace))
            except Exception as exc:
                failure=dict(split=split,controller=label,episode_id=episode,reason=f'{type(exc).__name__}: {exc}')
                failures.append(failure);rows.append(dict(**failure,failed=True))
                save(output/(episode+'_'+label.replace(' ','_')+'_failure.json'),failure)
        if (number+1)%10==0 or number+1==len(episodes):print(f'CLOSED LOOP {split}: {number+1}/{len(episodes)} episodes, four controllers; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'per_episode_metrics.csv',index=False)
    aggregate=[]
    for label,pairs in traces.items():
        if not pairs:continue
        reference=np.concatenate([data['reference_preview'][:,0] for data,trace in pairs])
        actual=np.concatenate([trace['cartesian'][1:] for data,trace in pairs]);errors=actual-reference
        distance=np.linalg.norm(errors[:,:2],axis=1)
        ep=frame[(frame.controller==label)&(~frame.failed)]
        metric=tracking_metrics(reference,actual,np.concatenate([data['reference_next_mechanical'] for data,trace in pairs]),
                                np.concatenate([trace['mechanical'][1:] for data,trace in pairs]),
                                np.concatenate([trace['commands'] for data,trace in pairs]),pairs[0][0]['previous_command'][0])
        # Command changes are averaged within episodes, never across reset boundaries.
        for key in ['mean_abs_delta_lift_V','mean_abs_delta_jib_V']:metric[key]=float(ep[key].mean())
        metric['maximum_command_change_V']=float(ep.maximum_command_change_V.max())
        for key in ['lift_command_delta_sign_changes','jib_command_delta_sign_changes']:metric[key]=int(ep[key].sum())
        metric['final_tip_error_mm']=float(ep.final_tip_error_mm.mean())
        metric.update(controller=label,episodes=len(pairs),samples=len(actual),failed_episodes=sum(f['controller']==label for f in failures),
                      median_episode_RMSE_mm=float(ep.tip_position_RMSE_mm.median()),p95_episode_RMSE_mm=float(ep.tip_position_RMSE_mm.quantile(.95)),
                      worst_episode_RMSE_mm=float(ep.tip_position_RMSE_mm.max()),
                      mean_solve_or_policy_ms=float(np.concatenate([trace['policy_call_ms'] for data,trace in pairs]).mean()),
                      p95_solve_or_policy_ms=float(np.percentile(np.concatenate([trace['policy_call_ms'] for data,trace in pairs]),95)),
                      deadline_miss_percent=float(np.mean(np.concatenate([trace['policy_call_ms'] for data,trace in pairs])>100)*100),
                      samples_outside_training_envelope_percent=float(ep.mechanical_samples_outside_training_envelope_percent.mean()))
        aggregate.append(metric)
    save(output/'aggregate_metrics.json',aggregate)
    return aggregate,frame,traces


def imitation(split):
    membership=pd.read_csv(ROOT/'data_splits/episode_membership.csv');episodes=membership[membership.split==split].episode_id.tolist()
    parts=[load_episode(episode) for episode in episodes];inputs=np.stack([data['supervised_inputs'] for data in parts])
    raw=np.stack([data['target_command'] for data in parts]);nominal=np.clip(raw,-1,1)
    mechanical=np.concatenate([data['current_mechanical'] for data in parts]);threshold=float(json.loads((base.STUDY/'low_velocity_definition.json').read_text())['maximum_boom_speed_rad_s'])
    masks=dict(nonsaturated_teacher=np.all(abs(raw.reshape(-1,2))<.95,axis=1),near_boundary_teacher=np.any(abs(raw.reshape(-1,2))>=.95,axis=1),
               low_velocity=np.max(abs(mechanical[:,2:4]),axis=1)<=threshold,moving=np.max(abs(mechanical[:,2:4]),axis=1)>threshold)
    result={};predictions={};seed_rows=[]
    settings=json.loads((ROOT/'configs/training.json').read_text())
    for architecture in ['mlp','gru']:
        policy=Policy(ROOT/'models'/architecture/'policy.pt');predicted=policy.predict_episodes(inputs)
        label='MLP' if architecture=='mlp' else 'GRU policy';predictions[label]=predicted
        result[label]=dict(original_actual_targets=command_metrics(raw.reshape(-1,2),predicted.reshape(-1,2)),
                           nominal_training_targets=command_metrics(nominal.reshape(-1,2),predicted.reshape(-1,2)),
                           subsets={name:command_metrics(raw.reshape(-1,2)[mask],predicted.reshape(-1,2)[mask]) if mask.any() else None for name,mask in masks.items()})
        for seed in settings['seeds']:
            model=Policy(ROOT/'models'/architecture/f'seed_{seed}'/'best_policy.pt')
            prediction=model.predict_episodes(inputs)
            seed_rows.append(dict(architecture=architecture,seed=seed,**command_metrics(raw.reshape(-1,2),prediction.reshape(-1,2))))
    save(ROOT/'results/imitation'/('imitation_metrics_'+split+'.json'),result)
    pd.DataFrame(seed_rows).to_csv(ROOT/'results/imitation'/('seed_metrics_'+split+'.csv'),index=False)
    variation=pd.DataFrame(seed_rows).groupby('architecture').combined_RMSE_V.agg(['mean','std','min','max']).reset_index()
    variation.to_csv(ROOT/'results/imitation'/('seed_variation_'+split+'.csv'),index=False)
    return result,predictions,parts


def benchmark():
    membership=pd.read_csv(ROOT/'data_splits/episode_membership.csv');validation=membership[membership.split=='validation'].episode_id.tolist()
    parts=[load_episode(episode)['supervised_inputs'] for episode in validation];inputs=np.concatenate(parts)
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    except OSError:pass
    results=[]
    for architecture in ['mlp','gru']:
        policy=Policy(ROOT/'models'/architecture/'policy.pt')
        for k in range(500):
            if k%50==0:policy.reset()
            x=inputs[k%len(inputs)];policy.step_components(x[:22],x[22:24],x[24:].reshape(2,4))
        latency=np.empty(10000)
        for k in range(10000):
            x=inputs[k%len(inputs)]
            begin=time.perf_counter_ns()
            if k%50==0:policy.reset()
            action=policy.step_components(x[:22],x[22:24],x[24:].reshape(2,4))
            latency[k]=(time.perf_counter_ns()-begin)/1e6
        label='MLP' if architecture=='mlp' else 'GRU policy'
        np.save(ROOT/'results/realtime'/(architecture+'_latency_ms.npy'),latency,allow_pickle=False)
        parameters=int(policy.bundle['parameter_count']);weight_bytes=sum(t.numel()*t.element_size() for t in policy.model.parameters())
        temporary_state_bytes=32*4+(64+64+32)*4 if architecture=='mlp' else 32*4+32*4+16*4
        macs=32*64+64*64+64*32+32*2 if architecture=='mlp' else 3*(32*32+32*32)+32*16+16*2
        result=dict(controller=label,calls=10000,warmup_calls=500,CPU=cpu,OS=platform.platform(),torch_threads=torch.get_num_threads(),
                    mean_ms=float(latency.mean()),median_ms=float(np.median(latency)),p95_ms=float(np.percentile(latency,95)),p99_ms=float(np.percentile(latency,99)),
                    maximum_ms=float(latency.max()),std_ms=float(latency.std()),deadline_ms=100.,
                    mean_real_time_factor_percent=float(latency.mean()/100*100),p95_real_time_factor_percent=float(np.percentile(latency,95)/100*100),
                    maximum_real_time_factor_percent=float(latency.max()/100*100),deadline_miss_percent=float(np.mean(latency>100)*100),
                    parameter_count=parameters,model_file_bytes=policy.path.stat().st_size,weight_storage_bytes=weight_bytes,
                    approximate_model_scaler_state_activation_bytes=weight_bytes+512+temporary_state_bytes,approximate_MACs_per_call=macs,
                    memory_note='Model/scaler/state/activation storage estimate; excludes Python/PyTorch framework RSS and allocator overhead.',
                    included='Input assembly and shape/finite checks, train-only normalization, CPU inference, own GRU state update/reset, output conversion and clipping.',
                    excluded='Simulated plant integration and frozen full-state observer propagation; state estimate is an input to the policy.')
        results.append(result);print('LATENCY '+json.dumps(result),flush=True)
    save(ROOT/'results/realtime/realtime_metrics.json',results)
    return results


def robustness(frame,traces,selected):
    rows=frame[(frame.controller==selected)&(~frame.failed)].copy()
    rows.sort_values('tip_position_RMSE_mm',ascending=False).head(5).to_csv(ROOT/'results/closed_loop/worst_episodes.csv',index=False)
    rows.groupby('source_recording').agg(episodes=('episode_id','count'),mean_episode_RMSE_mm=('tip_position_RMSE_mm','mean'),
                                       worst_episode_RMSE_mm=('tip_position_RMSE_mm','max')).to_csv(ROOT/'results/closed_loop/per_recording_robustness.csv')
    sets=[]
    low=float(json.loads((base.STUDY/'low_velocity_definition.json').read_text())['maximum_boom_speed_rad_s'])
    for data,trace in traces[selected]:
        mech=trace['mechanical'][1:];error=trace['cartesian'][1:]-data['reference_preview'][:,0]
        masks=dict(lift_positive=mech[:,2]>.001,lift_negative=mech[:,2]<-.001,jib_positive=mech[:,3]>.001,jib_negative=mech[:,3]<-.001,
                   low_velocity=np.max(abs(mech[:,2:]),axis=1)<=low,moving=np.max(abs(mech[:,2:]),axis=1)>low,
                   saturated=np.any(abs(trace['commands'])>=1-1e-6,axis=1),nonsaturated=np.all(abs(trace['commands'])<1-1e-6,axis=1))
        for category,mask in masks.items():
            if mask.any():sets.append(dict(category=category,samples=int(mask.sum()),position_SSE=float(np.sum(error[mask,:2]**2))))
    grouped=pd.DataFrame(sets).groupby('category').sum();grouped['position_RMSE_mm']=np.sqrt(grouped.position_SSE/grouped.samples)*1000
    grouped.to_csv(ROOT/'results/closed_loop/operation_category_robustness.csv')
    worst=rows.sort_values('tip_position_RMSE_mm',ascending=False).head(5).episode_id.tolist()
    out=ROOT/'results/closed_loop/worst_traces';out.mkdir(exist_ok=True)
    for data,trace in traces[selected]:
        if data['episode_id'][0] in worst:np.savez_compressed(out/(str(data['episode_id'][0])+'.npz'),**trace,reference=data['reference_preview'][:,0])


def evaluate():
    ctx=base.Teacher()
    policies={'MLP':Policy(ROOT/'models/mlp/policy.pt'),'GRU policy':Policy(ROOT/'models/gru/policy.pt')}
    # Validation alone selects the architecture; internal test cannot change weights/seeds.
    validation,_,_=run_closed_split(ctx,'validation',policies)
    eligible=[row for row in validation if row['controller'] in policies and row['failed_episodes']==0]
    chosen=min(eligible,key=lambda row:row['tip_position_RMSE_mm'])['controller'] if eligible else 'neither'
    frozen=dict(candidate_controller=chosen,selection_basis='Validation closed-loop tip RMSE; each seed/checkpoint already selected by validation MSE only.',
                validation_metrics=validation,MLP_sha256=digest(ROOT/'models/mlp/policy.pt'),GRU_sha256=digest(ROOT/'models/gru/policy.pt'),
                internal_test_not_used_for_selection=True,evaluation_models_frozen=True)
    save(ROOT/'models/architecture_selection.json',frozen)
    timing=benchmark()
    imitation_result,predictions,parts=imitation('internal_test')
    save(ROOT/'results/imitation/imitation_metrics.json',imitation_result)
    internal,frame,traces=run_closed_split(ctx,'internal_test',policies)
    save(ROOT/'results/closed_loop/closed_loop_metrics.json',dict(validation=validation,internal_test=internal,
         initial_state_only_from_dataset=True,measured_state_refresh_after_start=False,
         full_state_observer='Frozen GRU propagates nonmechanical estimates; Compact plant corrects mechanical fields after applying the policy action.',
         policy_uses_teacher_memory=False,teacher_baseline='Saved MPC trajectories and timings; no SLSQP solve or MPC regeneration performed.',
         recorded_replay_note='Oracle diagnostic: the reference was generated by this held command sequence, so replay error is zero by construction.'))
    if chosen!='neither':robustness(frame,traces,chosen)
    mpc=next(row for row in internal if row['controller']=='MPC')
    comparison=[]
    for row in internal:
        label=row['controller'];r=dict(row)
        if label in policies:
            lat=next(value for value in timing if value['controller']==label)
            r.update(mean_latency_ms=lat['mean_ms'],p95_latency_ms=lat['p95_ms'],p99_latency_ms=lat['p99_ms'],
                     deadline_miss_percent=lat['deadline_miss_percent'],parameters=lat['parameter_count'],
                     neural_RMSE_over_MPC_RMSE=row['tip_position_RMSE_mm']/mpc['tip_position_RMSE_mm'],
                     tracking_degradation_percent=(row['tip_position_RMSE_mm']/mpc['tip_position_RMSE_mm']-1)*100,
                     measured_speedup_vs_MPC=mpc['mean_solve_or_policy_ms']/lat['mean_ms'])
        else:r.update(mean_latency_ms=row['mean_solve_or_policy_ms'],p95_latency_ms=row['p95_solve_or_policy_ms'])
        comparison.append(r)
    pd.DataFrame(comparison).to_csv(ROOT/'results/final_comparison.csv',index=False)
    save(ROOT/'results/comparison/final_summary.json',dict(validation_selected_candidate=chosen,validation=validation,internal_test=comparison,
         imitation=imitation_result,realtime=timing,external_test_files_opened=0,external_test_files_modified=0,
         timing_limit='MPC timings are saved under eight-worker corpus generation; neural benchmarking is sequential/isolated, so speedups have different CPU-load contexts.',
         interpretation_pending='Read metrics before making the paper-readiness judgement; no arbitrary tracking pass/fail threshold was imposed.'))
    from policy_figures import make_figures
    make_figures(imitation_result,predictions,parts,frame,traces,timing,comparison)
    verify_immutable()
    text='NEURAL CONTROLLER EXPERIMENT METRICS COMPLETE\n\nValidation-selected candidate: '+chosen
    text+='\n\nInternal-test controller comparison:\n'+pd.DataFrame(comparison).to_string(index=False)
    text+='\n\nImitation:\n'+json.dumps(imitation_result,indent=2)+'\n\nReal-time:\n'+json.dumps(timing,indent=2)
    text+='\n\nORIGINAL TEST FILES OPENED: 0\nORIGINAL TEST FILES MODIFIED: 0\nNo MATLAB, commit or push performed.\n'
    (ROOT/'results/RESULTS.txt').write_text(text,encoding='utf-8');print(text,flush=True)
