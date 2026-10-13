import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib.metadata
import json
import sys
import time

from environment import ROOT,DATA_ROOT,audit,save,digest,snapshot,verify_immutable,load_episode,initial_rows,base

AUDIT_PLANT=None
AUDIT_CORE=None
AUDIT_MAPPING=None


def choose_sources(names,counts,target,preferred):
    states={(0,0):()}
    for name in names:
        for (size,number),chosen in list(states.items())[::-1]:
            key=(size+counts[name],number+1)
            if key not in states:states[key]=chosen+(name,)
    key=min((key for key in states if key[1]>0),key=lambda key:(abs(key[0]-target),abs(key[1]-preferred),key[1]))
    return states[key]


def prepare():
    import numpy as np
    import pandas as pd
    for folder in ['configs','data_splits','models/mlp','models/gru','results/imitation','results/closed_loop','results/realtime','results/comparison','figures','logs']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    if not (ROOT/'configs/accepted_sha256.json').exists():save(ROOT/'configs/accepted_sha256.json',snapshot())
    verify_immutable()
    if (ROOT/'data_splits/source_split.json').exists():return
    X=np.load(DATA_ROOT/'arrays/X_physical.npy',mmap_mode='r');Y=np.load(DATA_ROOT/'arrays/Y_command.npy',mmap_mode='r')
    ids=np.load(DATA_ROOT/'arrays/episode_id.npy');step=np.load(DATA_ROOT/'arrays/step.npy');sources=np.load(DATA_ROOT/'arrays/source_recording.npy')
    assert X.shape==(25000,32) and Y.shape==(25000,2)
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    assert len(set(zip(ids,step)))==25000
    manifest=pd.read_csv(DATA_ROOT/'manifests/episode_manifest.csv')
    counts=manifest.groupby('source_recording').size().to_dict()
    names=sorted(counts);rng=np.random.default_rng(20261012);rng.shuffle(names)
    validation=choose_sources(names,counts,75,6)
    remaining=[name for name in names if name not in validation]
    internal=choose_sources(remaining,counts,75,6)
    training=[name for name in names if name not in validation and name not in internal]
    partitions=dict(train=sorted(training),validation=sorted(validation),internal_test=sorted(internal))
    assert set(training).isdisjoint(validation) and set(training).isdisjoint(internal) and set(validation).isdisjoint(internal)
    indices={}
    for split,names in partitions.items():
        indices[split]=np.flatnonzero(np.isin(sources,names));np.save(ROOT/'data_splits'/(split+'_indices.npy'),indices[split],allow_pickle=False)
    membership=[]
    for row in manifest.to_dict('records'):
        split=next(split for split,names in partitions.items() if row['source_recording'] in names)
        membership.append(dict(episode_id=row['episode_id'],source_recording=row['source_recording'],split=split,samples=row['accepted_samples']))
        rows=np.flatnonzero(ids==row['episode_id']);assert len(rows)==50 and np.array_equal(step[rows],np.arange(50))
        assert np.all(np.isin(rows,indices[split]))
    pd.DataFrame(membership).to_csv(ROOT/'data_splits/episode_membership.csv',index=False)
    save(ROOT/'data_splits/source_split.json',dict(seed=20261012,method='Deterministic source-group subset assignment targeting 350/75/75 complete episodes; labels not used for split selection.',
         sources=partitions,sample_counts={split:len(index) for split,index in indices.items()},source_leakage_detected=False,episodes_intact=True))
    train=np.asarray(X[indices['train']],np.float64);mean=train.mean(0);std=train.std(0);effective=np.maximum(std,1e-6)
    save(ROOT/'configs/input_scaler.json',dict(fitted_split='train only',fitted_samples=len(train),fitted_sources=sorted(training),
          mean=mean.tolist(),raw_std=std.tolist(),effective_std=effective.tolist(),absolute_std_floor=1e-6,
          reason='Avoid amplifying effectively constant numerical noise; no validation/internal-test statistics used.'))
    # Preserve the original actual-applied teacher targets. A distinct nominal view
    # meets the policy interface's strict [-1,1] bounds, with documented microvolt differences.
    bounded=np.clip(np.asarray(Y),-1.,1.);np.save(ROOT/'data_splits/Y_command_nominal.npy',bounded,allow_pickle=False)
    save(ROOT/'data_splits/target_projection.json',dict(original_unchanged=True,raw_targets_are_actual_teacher_commands=True,
         reason='Neural interface has exact nominal [-1,1] limits; frozen teacher bounds have tiny numerical excess. This is a documented projection, not a claim the teacher plant clipped its input.',
         changed_values=int(np.sum(bounded!=Y)),max_absolute_change_V=float(np.max(abs(bounded-Y))),
         original_target_path=str(DATA_ROOT/'arrays/Y_command.npy'),derived_target_path=str(ROOT/'data_splits/Y_command_nominal.npy'),
         imitation_metrics_include_original_actual_targets=True))
    save(ROOT/'configs/training.json',dict(seeds=[42,123,2026],max_epochs=300,patience=30,learning_rate=.001,weight_decay=.0001,
         optimizer='AdamW',loss='physical command MSE',mlp_batch_samples=1024,gru_batch_episodes=32,
         architectures=dict(mlp=[32,64,64,32,2],gru=dict(input=32,hidden=32,head=[16,2])),
         output='tanh plus final [-1,1] clipping',teacher_memory_supervised=False,epoch_selection='validation command MSE',
         seed_selection='validation command MSE',architecture_selection='validation closed-loop tracking, before internal-test evaluation',
         internal_test_tuning=False,hyperparameter_search=False))
    save(ROOT/'configs/runtime_versions.json',{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','torch','numba','matplotlib','pyarrow']})
    print('SOURCE-GROUP SPLIT:',{split:len(index) for split,index in indices.items()},'Source leakage: NO',flush=True)


def audit_init():
    global AUDIT_PLANT,AUDIT_CORE,AUDIT_MAPPING
    sys.addaudithook(audit)
    import torch
    torch.set_num_threads(2)
    import offline_mpc as core
    AUDIT_CORE=core
    AUDIT_PLANT=core.MechanicalPlant()
    assert sum(p.numel() for p in AUDIT_PLANT.model.parameters())==1060
    AUDIT_MAPPING=[core.INPUT_COLS.index(name) for name in core.PLANT_INPUT_COLS]


def audit_episode(episode_id):
    import numpy as np
    data=load_episode(episode_id)
    assert data['supervised_inputs'].shape==(50,32) and data['target_command'].shape==(50,2)
    assert np.array_equal(data['supervised_inputs'][:,:22],data['full_state'])
    assert np.array_equal(data['supervised_inputs'][:,22:24],data['previous_command'])
    assert np.array_equal(data['supervised_inputs'][:,24:],data['reference_preview'].reshape(50,8))
    assert np.array_equal(data['step'],np.arange(50)) and np.all(np.diff(data['source_time_s'])>0)
    assert np.array_equal(data['previous_command'][1:],data['target_command'][:-1])
    for key,value in data.items():
        if value.dtype.kind in 'fi':assert np.isfinite(value).all(),key
    initial=initial_rows(data,type('Context',(),{'core':AUDIT_CORE})())
    AUDIT_PLANT.reset(initial[:,AUDIT_MAPPING])
    maximum=0.
    for k,action in enumerate(data['target_command']):
        actual=AUDIT_PLANT.advance(action,internal_steps=10000)
        maximum=max(maximum,float(np.max(abs(actual-data['next_mechanical'][k]))))
    assert maximum<1e-7,f'Applied-target/plant mismatch in {episode_id}: {maximum}'
    return dict(episode_id=episode_id,max_replayed_mechanical_difference=maximum,actual_applied_targets_verified=True)


def audit_corpus():
    import numpy as np
    import pandas as pd
    if (ROOT/'results/comparison/dataset_audit.json').exists():
        print((ROOT/'results/comparison/dataset_audit.json').read_text(),flush=True);return
    inventory=json.loads((DATA_ROOT/'metadata/corpus_sha256.json').read_text())
    for relative,sha in inventory.items():assert digest(DATA_ROOT/relative)==sha,relative
    membership=pd.read_csv(ROOT/'data_splits/episode_membership.csv');episodes=membership.episode_id.tolist()
    results=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,initializer=audit_init) as pool:
        jobs=[pool.submit(audit_episode,episode) for episode in episodes]
        for job in as_completed(jobs):
            results.append(job.result())
            if len(results)%50==0:print(f'Applied-command audit {len(results)}/500; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    Y=np.load(DATA_ROOT/'arrays/Y_command.npy')
    config=json.loads((DATA_ROOT/'configs/frozen_teacher.json').read_text());low,high=np.asarray(config['command_bounds']['low']),np.asarray(config['command_bounds']['high'])
    assert np.all(Y>=low-1e-9) and np.all(Y<=high+1e-9)
    report=dict(status='PASS',input_shape=[25000,32],output_shape=[25000,2],finite_samples=25000,episodes=500,
                chronological_steps=True,duplicate_episode_steps=0,source_leakage_detected=False,
                actual_applied_targets_verified=True,max_plant_replay_difference=max(r['max_replayed_mechanical_difference'] for r in results),
                within_frozen_teacher_bounds=True,nominal_bound_excess_values=int(np.sum(abs(Y)>1)),
                nominal_bound_max_excess_V=float(np.maximum(abs(Y)-1,0).max()),
                neural_target_view='Documented nominal [-1,1] projection; raw applied teacher targets retained for diagnostics and imitation scoring',
                teacher_dataset_modified=False,external_test_files_opened=0,external_test_files_modified=0)
    save(ROOT/'results/comparison/dataset_audit.json',report);pd.DataFrame(results).to_csv(ROOT/'results/comparison/applied_command_replay_audit.csv',index=False)
    verify_immutable();print('DATASET AUDIT BEFORE TRAINING\n'+json.dumps(report,indent=2),flush=True)


def experiment_main():
    parser=argparse.ArgumentParser(description='Internal neural policy training and evaluation; original Test remains blocked')
    parser.add_argument('--stage',choices=['all','audit','train','evaluate'],default='all')
    args=parser.parse_args();sys.addaudithook(audit)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    prepare()
    if args.stage in ['all','audit','train']:audit_corpus()
    if args.stage=='audit':return
    if args.stage in ['all','train']:
        from train_policies import train_all
        train_all()
    if args.stage in ['all','evaluate']:
        from evaluate_policies import evaluate
        evaluate()
    verify_immutable()


if __name__=='__main__':main()
