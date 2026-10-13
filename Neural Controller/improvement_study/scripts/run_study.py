import sys
sys.dont_write_bytecode = True
from pathlib import Path
import os
BASE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('NEURAL_IMPROVEMENT_OUTPUT_DIR', BASE.parent/'results/neural_improvement')).resolve()
sys.path.insert(0, str(BASE/'scripts'))
import environment as env
import json, os, random, time
import numpy as np
import pandas as pd
import torch
from policy_runtime import Policy, make_model
from evaluate_policies import closed_episode, teacher_trace, tracking_metrics, command_metrics

def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    env.save(path, value)

def guard(event, args):
    env.audit(event, args)
    if event not in ['open','os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.utime']: return
    for value in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(value,(str,bytes,os.PathLike)): continue
        p=Path(os.fsdecode(value)).resolve()
        if not p.is_relative_to(BASE) or p.is_relative_to(ROOT): continue
        write=event!='open'
        if event=='open':
            mode,flags=args[1:3]
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write: raise PermissionError(f'Frozen baseline: {p}')

def baseline_snapshot():
    return {str(p.relative_to(BASE)):env.digest(p) for p in BASE.rglob('*') if not p.is_relative_to(ROOT) and p.is_file()}

def dynamics_snapshot():
    project=BASE.parent
    paths=[project/'GRU Latent/artifacts/best_model.pt',project/'MPC offline Model/artifacts/mechanical_plant/best_model.pt',project/'MPC offline Model/configs/final_gru_mpc.json']
    return {str(p):env.digest(p) for p in paths}

def split_parts(split):
    membership=pd.read_csv(BASE/'data_splits/episode_membership.csv')
    return [env.load_episode(ep) for ep in membership[membership.split==split].episode_id]

def arrays(parts):
    return np.stack([p['supervised_inputs'] for p in parts]),np.stack([p['target_command'] for p in parts])

def weights(y, threshold):
    delta=np.zeros(y.shape[:2]); delta[:,1:]=np.max(abs(np.diff(y,axis=1)),axis=2)
    return 1+(np.max(abs(y),axis=2)>=.8).astype(float)+(delta>=threshold).astype(float)

def train():
    settings=read(BASE/'configs/training.json');scaler=read(BASE/'configs/input_scaler.json')
    tx,ty=arrays(split_parts('train')); vx,vy=arrays(split_parts('validation'))
    master_x=np.load(env.DATA_ROOT/'arrays/X_physical.npy',mmap_mode='r')
    nominal_y=np.load(env.DATA_ROOT/'arrays/Y_command_nominal.npy',mmap_mode='r')
    for split,a,b in [('train',tx,ty),('validation',vx,vy)]:
        indices=np.load(BASE/'data_splits'/f'{split}_indices.npy')
        assert np.array_equal(a.reshape(-1,32),master_x[indices]),'Episode order differs from frozen indices'
        assert np.array_equal(np.clip(b,-1,1).reshape(-1,2),nominal_y[indices]),'Target view differs from baseline'
    threshold=float(np.percentile(np.max(abs(np.diff(ty,axis=1)),axis=2),75))
    save(ROOT/'config/weighting.json',dict(boundary_V=.8,transition_percentile=75,transition_threshold_V=threshold,
         rule='1 + any(abs(raw_teacher)>=0.8) + max(abs(within-episode raw teacher delta))>=train p75',maximum_weight=3,
         first_sample_transition_weight=False,threshold_partition='train only',delta_loss='unweighted MSE of k=1..49; no cross-episode delta'))
    mean,std=np.asarray(scaler['mean']),np.asarray(scaler['effective_std'])
    tx=torch.tensor((tx-mean)/std,dtype=torch.float32);vx=torch.tensor((vx-mean)/std,dtype=torch.float32)
    tw=torch.tensor(weights(ty,threshold),dtype=torch.float32);vw=torch.tensor(weights(vy,threshold),dtype=torch.float32)
    ty=torch.tensor(np.clip(ty,-1,1),dtype=torch.float32);vy=torch.tensor(np.clip(vy,-1,1),dtype=torch.float32)
    feature_order=[f['name'] for f in read(env.FEATURE_SCHEMA)['supervised_inputs']]
    runs=[]
    for lam in [0.,.05,.1,.2]:
        for seed in settings['seeds']:
            name=f'weighted_delta_{lam:g}_seed_{seed}'; folder=ROOT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists(): runs.append(read(folder/'complete.json'));continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=make_model('gru');assert sum(p.numel() for p in model.parameters())==6898
            opt=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=10,min_lr=1e-5)
            best=float('inf'); best_plain=float('inf'); stale=0; history=[]
            def objective(pred,target,w):
                primary=((pred-target).square().mean(-1)*w).sum()/w.sum()
                temporal=(torch.diff(pred,dim=1)-torch.diff(target,dim=1)).square().mean()
                return primary+lam*temporal
            for epoch in range(1,settings['max_epochs']+1):
                model.train();order=torch.randperm(len(tx));total=0
                for start in range(0,len(tx),settings['gru_batch_episodes']):
                    ids=order[start:start+settings['gru_batch_episodes']];opt.zero_grad(set_to_none=True)
                    pred,_=model(tx[ids]);loss=objective(pred,ty[ids],tw[ids]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(ids)
                model.eval()
                with torch.inference_mode():
                    pred,_=model(vx);val=float(objective(pred,vy,vw));plain=float((pred-vy).square().mean())
                sched.step(val);history.append(dict(epoch=epoch,train_loss=total/len(tx),validation_loss=val,validation_MSE=plain))
                bundle=dict(format='neural_crane_policy_v1',architecture='gru',weights=model.state_dict(),scaler=scaler,
                            feature_order=feature_order,parameter_count=6898,seed=seed,best_epoch=epoch,lambda_delta=lam,
                            output_bounds=[-1.,1.],teacher_memory_is_input=False)
                if val<best-1e-7:
                    best=val;stale=0;torch.save(bundle,folder/'best_objective.pt')
                else: stale+=1
                if plain<best_plain-1e-7:
                    best_plain=plain;torch.save(bundle,folder/'best_command.pt')
                if epoch%20==0 or stale>=settings['patience']:
                    print(f'TRAIN {name} epoch={epoch} validation command RMSE={np.sqrt(plain):.6f} V',flush=True)
                    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
                if stale>=settings['patience']: break
            pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            summary=dict(name=name,seed=seed,lambda_delta=lam,epochs=epoch,best_validation_loss=best)
            save(folder/'complete.json',summary);runs.append(summary)
    save(ROOT/'results/comparison/training_runs.json',runs)

def closed(ctx, parts, checkpoint, folder):
    folder.mkdir(parents=True,exist_ok=True);policy=Policy(checkpoint);pairs=[];rows=[]
    for i,data in enumerate(parts):
        ep=str(data['episode_id'][0]);path=folder/(ep+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as f: trace={k:f[k] for k in f.files}
        else:
            trace=closed_episode(ctx,data,policy);np.savez_compressed(path,**trace)
        met=tracking_metrics(data['reference_preview'][:,0],trace['cartesian'][1:],data['reference_next_mechanical'],trace['mechanical'][1:],trace['commands'],data['previous_command'][0])
        rows.append(dict(episode_id=ep,source_recording=str(data['source_recording'][0]),**met));pairs.append((data,trace))
        if (i+1)%5==0:print(f'CLOSED {folder.name}: {i+1}/{len(parts)}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(folder/'per_episode_metrics.csv',index=False)
    return aggregate(pairs,frame),frame,pairs

def aggregate(pairs,frame):
    metric=tracking_metrics(np.concatenate([d['reference_preview'][:,0] for d,t in pairs]),np.concatenate([t['cartesian'][1:] for d,t in pairs]),
        np.concatenate([d['reference_next_mechanical'] for d,t in pairs]),np.concatenate([t['mechanical'][1:] for d,t in pairs]),
        np.concatenate([t['commands'] for d,t in pairs]),pairs[0][0]['previous_command'][0])
    for key in ['mean_abs_delta_lift_V','mean_abs_delta_jib_V','final_tip_error_mm']:metric[key]=float(frame[key].mean())
    for key in ['lift_command_delta_sign_changes','jib_command_delta_sign_changes']:metric[key]=int(frame[key].sum())
    delta=np.concatenate([np.diff(np.vstack((d['previous_command'][0],t['commands'])),axis=0) for d,t in pairs])
    commands=np.concatenate([t['commands'] for d,t in pairs])
    for j,name in enumerate(['lift','jib']):
        metric[f'p95_abs_delta_{name}_V']=float(np.percentile(abs(delta[:,j]),95))
        metric[f'maximum_abs_delta_{name}_V']=float(abs(delta[:,j]).max())
        changes=sum(int(np.sum(t['commands'][1:,j]*t['commands'][:-1,j]<0)) for d,t in pairs)
        metric[f'{name}_command_sign_changes']=changes
        metric[f'{name}_command_sign_changes_per_second']=changes/(len(pairs)*4.9)
    metric['maximum_command_change_V']=float(abs(delta).max())
    metric.update(episodes=len(pairs),samples=len(commands),failed_episodes=0,nonfinite_episodes=0,
        median_episode_RMSE_mm=float(frame.tip_position_RMSE_mm.median()),p95_episode_RMSE_mm=float(frame.tip_position_RMSE_mm.quantile(.95)),
        worst_episode_RMSE_mm=float(frame.tip_position_RMSE_mm.max()))
    return metric

def validation_worker(run):
    sys.addaudithook(guard);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    parts=split_parts('validation');ctx=env.base.Teacher()
    grouped={}
    for d in parts:grouped.setdefault(str(d['source_recording'][0]),[]).append(d)
    screen=[d for group in grouped.values() for d in group[:2]]
    name=run['name'];folder=ROOT/'models'/name;screen_rows=[]
    for checkpoint in ['best_objective.pt','best_command.pt']:
        metrics,_,_=closed(ctx,screen,folder/checkpoint,ROOT/'results/closed_loop/screen'/name/checkpoint.replace('.pt',''))
        screen_rows.append(dict(checkpoint=checkpoint,**metrics))
    chosen=min(screen_rows,key=lambda r:r['tip_position_RMSE_mm'])['checkpoint']
    metrics,_,_=closed(ctx,parts,folder/chosen,ROOT/'results/closed_loop/validation'/name)
    x,y=arrays(parts);pred=Policy(folder/chosen).predict_episodes(x)
    row=dict(**run,checkpoint=str(folder/chosen),validation_command_RMSE_V=command_metrics(y.reshape(-1,2),pred.reshape(-1,2))['combined_RMSE_V'],**metrics)
    save(ROOT/'results/comparison'/(name+'_validation.json'),row)
    print(f'VALIDATION {name}: tip RMSE={metrics["tip_position_RMSE_mm"]:.6f} mm',flush=True)
    return row

def select(ctx):
    import subprocess
    parts=split_parts('validation');grouped={}
    for d in parts:grouped.setdefault(str(d['source_recording'][0]),[]).append(d)
    screen=[d for group in grouped.values() for d in group[:2]]
    save(ROOT/'config/selection_protocol.json',dict(screen_episode_ids=[str(d['episode_id'][0]) for d in screen],
         screening='Two fixed validation episodes per source, two loss-selected checkpoints per seed/config; checkpoint chosen by closed-loop tip RMSE.',
         final_selection='All 75 validation episodes select seed and lambda by closed-loop tip RMSE; ties use imitation RMSE.',internal_test_used=False))
    rows=[];pending=list(read(ROOT/'results/comparison/training_runs.json'));active=[]
    while pending or active:
        while pending and len(active)<3:
            run=pending.pop(0);log=(ROOT/'logs'/(run['name']+'.log')).open('w',encoding='utf-8')
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--validation-worker',run['name']],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active.append((process,log,run))
        for process,log,run in list(active):
            code=process.poll()
            if code is None:continue
            log.close();active.remove((process,log,run))
            if code:raise RuntimeError(f'Validation worker failed: {run["name"]}; see logs')
            row=read(ROOT/'results/comparison'/(run['name']+'_validation.json'));rows.append(row)
            save(ROOT/'results/comparison/validation_candidates.json',rows)
            print(f'VALIDATION {run["name"]}: tip RMSE={row["tip_position_RMSE_mm"]:.6f} mm',flush=True)
        time.sleep(.1)
    pd.DataFrame(rows).to_csv(ROOT/'results/comparison/validation_candidates.csv',index=False)
    winner=min(rows,key=lambda r:(r['tip_position_RMSE_mm'],r['validation_command_RMSE_V']))
    (ROOT/'models/selected_improved_gru.pt').write_bytes(Path(winner['checkpoint']).read_bytes())
    save(ROOT/'models/selection.json',dict(**winner,sha256=env.digest(ROOT/'models/selected_improved_gru.pt'),frozen_before_internal_test=True))

def timing(policy, inputs):
    for k in range(500):
        if k%50==0:policy.reset()
        x=inputs[k%len(inputs)];policy.step_components(x[:22],x[22:24],x[24:].reshape(2,4))
    latency=np.empty(10000)
    for k in range(10000):
        x=inputs[k%len(inputs)];start=time.perf_counter_ns()
        if k%50==0:policy.reset()
        policy.step_components(x[:22],x[22:24],x[24:].reshape(2,4));latency[k]=(time.perf_counter_ns()-start)/1e6
    return latency,latency_metrics(latency)

def latency_metrics(t):
    return dict(calls=len(t),mean_ms=float(t.mean()),median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),p99_ms=float(np.percentile(t,99)),maximum_ms=float(t.max()),deadline_miss_percent=float(np.mean(t>100)*100))

def finish(ctx):
    parts=split_parts('internal_test');x,y=arrays(parts);policies={'Baseline GRU':Policy(BASE/'models/gru/policy.pt'),'Improved GRU':Policy(ROOT/'models/selected_improved_gru.pt')}
    pred={label:p.predict_episodes(x) for label,p in policies.items()};threshold=read(ROOT/'config/weighting.json')['transition_threshold_V']
    delta=np.zeros(y.shape[:2]);delta[:,1:]=np.max(abs(np.diff(y,axis=1)),axis=2)
    speed=np.concatenate([d['current_mechanical'][:,2:] for d in parts]);low=read(env.base.STUDY/'low_velocity_definition.json')['maximum_boom_speed_rad_s']
    masks=dict(nonsaturated=np.all(abs(y.reshape(-1,2))<1-1e-6,axis=1),near_boundary=np.any(abs(y.reshape(-1,2))>=.8,axis=1),
               large_transition=delta.reshape(-1)>=threshold,low_velocity=np.max(abs(speed),axis=1)<=low,moving=np.max(abs(speed),axis=1)>low)
    imitation={label:dict(overall=command_metrics(y.reshape(-1,2),p.reshape(-1,2)),subsets={n:command_metrics(y.reshape(-1,2)[m],p.reshape(-1,2)[m]) if m.any() else None for n,m in masks.items()}) for label,p in pred.items()}
    save(ROOT/'results/imitation_metrics.json',imitation)
    save(ROOT/'results/imitation/imitation_metrics.json',imitation)
    save(ROOT/'results/imitation/subset_definitions.json',dict(nonsaturated='Both actual teacher commands have magnitude < 1 - 1e-6 V',near_boundary='Either command magnitude >= 0.8 V; may overlap nonsaturated',large_transition_train_threshold_V=threshold,low_velocity_max_boom_speed_rad_s=low))
    improved,iframe,ipairs=closed(ctx,parts,ROOT/'models/selected_improved_gru.pt',ROOT/'results/closed_loop/internal_test')
    pairs={'Improved GRU':ipairs};frames={'Improved GRU':iframe};metrics={'Improved GRU':improved}
    for label,filelabel in [('Baseline GRU','GRU_policy'),('MPC','MPC')]:
        ps=[];rs=[]
        for d in parts:
            ep=str(d['episode_id'][0])
            with np.load(BASE/'results/closed_loop/internal_test'/(ep+'_'+filelabel+'.npz'),allow_pickle=False) as f:t={k:f[k] for k in f.files}
            ps.append((d,t));rs.append(dict(episode_id=ep,source_recording=str(d['source_recording'][0]),**tracking_metrics(d['reference_preview'][:,0],t['cartesian'][1:],d['reference_next_mechanical'],t['mechanical'][1:],t['commands'],d['previous_command'][0])))
        frames[label]=pd.DataFrame(rs);pairs[label]=ps;metrics[label]=aggregate(ps,frames[label])
    save(ROOT/'results/closed_loop_metrics.json',metrics)
    validation=np.concatenate([d['supervised_inputs'] for d in split_parts('validation')]);realtime={};latencies={}
    for label,policy in policies.items():
        lat,met=timing(policy,validation);latencies[label]=lat;np.save(ROOT/'results/realtime'/(label.replace(' ','_')+'_latency_ms.npy'),lat)
        loops=np.concatenate([t['policy_call_ms']+t['observer_plus_simulated_plant_ms'] for d,t in pairs[label]])
        realtime[label]=dict(**met,warmup_calls=500,torch_threads=2,parameter_count=6898,simulation_loop=latency_metrics(loops),
            included='Assembly, checks, train-only normalization, recurrent-state reset/update, CPU inference, conversion, clipping',simulation_loop_context='Saved baseline loop vs new improved loop; different run dates')
    save(ROOT/'results/realtime/archived_baseline_realtime_metrics.json',next(r for r in read(BASE/'results/realtime/realtime_metrics.json') if r['controller']=='GRU policy'))
    realtime['MPC']=latency_metrics(np.concatenate([t['policy_call_ms'] for d,t in pairs['MPC']]))
    save(ROOT/'results/realtime_metrics.json',realtime)
    source=pd.DataFrame({label:frame.groupby('source_recording').tip_position_RMSE_mm.apply(lambda v:float(np.sum(v*v))) for label,frame in frames.items()})
    counts=iframe.groupby('source_recording').size().reindex(source.index).to_numpy();rng=np.random.default_rng(20261012)
    boot={n:[] for n in ['improved_minus_baseline_mm','improved_minus_MPC_mm']}
    for k in range(2000):
        ids=rng.integers(0,len(source),len(source));rm={label:np.sqrt(source[label].to_numpy()[ids].sum()/counts[ids].sum()) for label in frames}
        boot['improved_minus_baseline_mm'].append(rm['Improved GRU']-rm['Baseline GRU']);boot['improved_minus_MPC_mm'].append(rm['Improved GRU']-rm['MPC'])
    bootstrap={n:dict(lower_95=float(np.percentile(v,2.5)),upper_95=float(np.percentile(v,97.5))) for n,v in boot.items()}
    bootstrap.update(resamples=2000,seed=20261012,clusters=len(source));save(ROOT/'results/bootstrap_metrics.json',bootstrap)
    b=metrics['Baseline GRU']['tip_position_RMSE_mm'];v=improved['tip_position_RMSE_mm'];m=metrics['MPC']['tip_position_RMSE_mm']
    replace=v<b and bootstrap['improved_minus_baseline_mm']['upper_95']<0 and improved['worst_episode_RMSE_mm']<=metrics['Baseline GRU']['worst_episode_RMSE_mm'] and realtime['Improved GRU']['deadline_miss_percent']==0
    decision='REPLACE WITH IMPROVED GRU' if replace else 'KEEP BASELINE'
    summary=dict(improved=v<b,improvement_percent=(1-v/b)*100,improved_over_baseline=v/b,improved_over_MPC=v/m,degradation_vs_MPC_percent=(v/m-1)*100,decision=decision,
        decision_rule='Lower tracking RMSE, source-cluster CI below zero, no worse worst episode, no failures/nonfinite, no deadline misses; conservative recommendation only.',external_Test_files_opened=0,external_Test_files_modified=0)
    save(ROOT/'results/comparison/decision.json',summary)
    rows=[]
    for label in ['MPC','Baseline GRU','Improved GRU']:
        rows.append(dict(controller=label,parameters=4098 if label=='MPC' else 6898,**metrics[label],command_RMSE_V=0 if label=='MPC' else imitation[label]['overall']['combined_RMSE_V'],**realtime[label]))
    flat=[{k:v for k,v in row.items() if not isinstance(v,dict)} for row in rows]
    table=pd.DataFrame(flat);table.to_csv(ROOT/'results/final_improvement_comparison.csv',index=False)
    from figures import make_figures
    make_figures(ROOT,parts,pred,pairs,frames,metrics,latencies,realtime,masks)
    text='FOCUSED GRU IMPROVEMENT STUDY\n\n'+table.to_string(index=False)+'\n\nImitation:\n'+json.dumps(imitation,indent=2)+'\n\nClosed-loop:\n'+json.dumps(metrics,indent=2)+'\n\nRealtime:\n'+json.dumps(realtime,indent=2)+'\n\nBootstrap:\n'+json.dumps(bootstrap,indent=2)+'\n\nDecision:\n'+json.dumps(summary,indent=2)
    text+='\n\nMPC timings are archived eight-worker generation timings, not matched-load architecture benchmarks.\nRollout fine-tuning skipped: frozen deployed plant is NumPy/Numba and non-differentiable.\nBaseline/corpus unchanged; no external Test, MATLAB, Git commit or push.\nOutput: '+str(ROOT)+'\n'
    (ROOT/'results/RESULTS.txt').write_text(text,encoding='utf-8');print(text,flush=True)

def main():
    sys.addaudithook(guard);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    env.verify_immutable()
    for folder in ['config','models','logs','figures','results/imitation','results/closed_loop','results/realtime','results/comparison']: (ROOT/folder).mkdir(parents=True,exist_ok=True)
    snapshot=baseline_snapshot()
    if (ROOT/'config/baseline_sha256.json').exists():assert snapshot==read(ROOT/'config/baseline_sha256.json'),'Baseline differs from original study snapshot'
    else:save(ROOT/'config/baseline_sha256.json',snapshot)
    dynamics=dynamics_snapshot()
    if (ROOT/'config/frozen_dynamics_sha256.json').exists():assert dynamics==read(ROOT/'config/frozen_dynamics_sha256.json'),'Frozen dynamics changed'
    else:save(ROOT/'config/frozen_dynamics_sha256.json',dynamics)
    membership=pd.read_csv(BASE/'data_splits/episode_membership.csv')
    sets={s:set(membership[membership.split==s].source_recording) for s in ['train','validation','internal_test']}
    assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','internal_test'),('validation','internal_test')])
    assert [len(np.load(BASE/'data_splits'/f'{s}_indices.npy')) for s in ['train','validation','internal_test']]==[17500,3750,3750]
    save(ROOT/'config/study.json',dict(source_leakage=0,split_counts=[17500,3750,3750],seeds=[42,123,2026],lambda_delta=[0,.05,.1,.2],parameter_count=6898,architecture_changed=False,
         optimizer=read(BASE/'configs/training.json'),baseline_checkpoint_sha256=env.digest(BASE/'models/gru/policy.pt'),scaler_sha256=env.digest(BASE/'configs/input_scaler.json'),rollout_fine_tuning='Skipped: frozen deployed plant non-differentiable NumPy/Numba'))
    train();ctx=env.base.Teacher()
    if not (ROOT/'models/selection.json').exists():select(ctx)
    finish(ctx)
    assert baseline_snapshot()==snapshot,'Baseline changed'
    assert dynamics_snapshot()==dynamics,'Frozen dynamics or MPC configuration changed'
    env.verify_immutable();save(ROOT/'results/comparison/access_audit.json',dict(baseline_unchanged=True,corpus_unchanged=True,prior_studies_unchanged=True,source_leakage=0,external_Test_files_opened=0,external_Test_files_modified=0))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Reproduce the focused weighted/delta GRU ablation; shipped baseline and results remain read-only')
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    parser.add_argument('--validation-worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    ROOT=args.output_dir.resolve()
    if ROOT.is_relative_to(BASE):parser.error('Choose an output outside the released Neural Controller directory')
    os.environ['NEURAL_IMPROVEMENT_OUTPUT_DIR']=str(ROOT)
    if args.validation_worker:
        run=next(r for r in read(ROOT/'results/comparison/training_runs.json') if r['name']==args.validation_worker);validation_worker(run)
    else:main()
