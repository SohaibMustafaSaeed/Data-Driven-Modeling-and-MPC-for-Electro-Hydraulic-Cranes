import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

RELEASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RELEASE.parent))
ENGINE=None
OUTPUT=None

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def init(output):
    global ENGINE,OUTPUT
    import teacher_engine as engine
    sys.addaudithook(engine.audit);engine.verify_frozen_configuration();ENGINE=engine.Teacher();OUTPUT=Path(output)

def episode(record):
    import numpy as np
    import teacher_engine as engine
    source=ENGINE.records[record['source_recording']]
    assert source['training_end_exclusive']==record['training_end_exclusive']
    record=dict(record,prepared_array=source['prepared_array'])
    path=OUTPUT/'episodes'/(record['episode_id']+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:data={k:saved[k] for k in saved.files}
    else:
        data=engine.episode_arrays(ENGINE,record)
        data['source_time_s']=data['csv_time_s'].copy();data['dataset_version']=np.full(50,'mpc_teacher_v1')
        x=ENGINE.data(record);ids=record['first_retained_row']+np.arange(50)*10000
        data['recorded_command']=np.asarray(x[ids][:,ENGINE.core.CI],float)*ENGINE.std[ENGINE.core.CI]+ENGINE.mean[ENGINE.core.CI]
        engine.validate_arrays(data,ENGINE)
        tmp=path.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,**data)
        tmp.replace(path)
    engine.validate_arrays(data,ENGINE)
    assert set(data['episode_id'])=={record['episode_id']} and set(data['source_recording'])=={record['source_recording']}
    return record['episode_id']

def main():
    parser=argparse.ArgumentParser(description='Regenerate exactly 500 frozen training-only episodes; never samples external Test')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--preprocessed-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path(os.environ.get('MPC_TEACHER_DATASET_DIR',RELEASE/'dataset/generated')))
    parser.add_argument('--plan-only',action='store_true',help='Validate the bundled manifest/config/schema without loading recordings or running MPC')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if args.data_root:os.environ['IMOTION_DATA_ROOT']=str(args.data_root.resolve())
    if args.preprocessed_dir:os.environ['IMOTION_PREPROCESSED_DIR']=str(args.preprocessed_dir.resolve())
    manifest=json.loads((RELEASE/'dataset/manifests/planned_episode_manifest.json').read_text())
    schema=json.loads((RELEASE/'dataset/metadata/feature_schema.json').read_text())
    assert len(manifest)==500 and len({r['episode_id'] for r in manifest})==500
    assert len({r['source_recording'] for r in manifest})==43 and schema['supervised_input_dimension']==32
    import teacher_engine as engine
    engine.verify_frozen_configuration()
    print('Frozen plan: 500 episodes x 50 updates = 25,000 samples; 43 Train recordings; 32 inputs / 2 targets.')
    if args.plan_only:return
    sys.addaudithook(engine.audit)
    output=args.output_dir.resolve()
    if output.is_relative_to(RELEASE) and not output.is_relative_to(RELEASE/'dataset/generated'):parser.error('Cannot overwrite released evidence')
    if output.exists() and any(output.iterdir()) and not (output/'metadata/regeneration.json').exists():parser.error('Nonempty output is not a resumable teacher regeneration directory; choose a fresh --output-dir')
    for name in ['episodes','arrays','metadata','summaries']:(output/name).mkdir(parents=True,exist_ok=True)
    import shutil
    (output/'config').mkdir(exist_ok=True)
    shutil.copy2(RELEASE/'dataset/metadata/frozen_teacher.json',output/'config/frozen_teacher.json')
    provenance={'manifest_sha256':hashlib.sha256((RELEASE/'dataset/manifests/planned_episode_manifest.json').read_bytes()).hexdigest(),
                'teacher_sha256':hashlib.sha256((RELEASE/'dataset/metadata/frozen_teacher.json').read_bytes()).hexdigest(),
                'schema_sha256':hashlib.sha256((RELEASE/'dataset/metadata/feature_schema.json').read_bytes()).hexdigest(),'dataset_version':'mpc_teacher_v1'}
    if (output/'metadata/regeneration.json').exists():assert json.loads((output/'metadata/regeneration.json').read_text())==provenance
    else:save(output/'metadata/regeneration.json',provenance)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(output),)) as pool:
        for count,future in enumerate(as_completed([pool.submit(episode,r) for r in manifest]),1):
            future.result();print(f'Complete {count}/500 episodes',flush=True)
    import numpy as np
    parts=[]
    for record in manifest:
        with np.load(output/'episodes'/(record['episode_id']+'.npz'),allow_pickle=False) as f:parts.append({k:f[k] for k in f.files})
    for source,name in [('supervised_inputs','X_physical'),('target_command','Y_command'),('episode_id','episode_id'),('step','step'),('source_recording','source_recording')]:
        array=np.concatenate([p[source] for p in parts]);np.save(output/'arrays'/(name+'.npy'),array,allow_pickle=False)
    y=np.concatenate([p['target_command'] for p in parts]);np.save(output/'arrays/Y_command_nominal.npy',np.clip(y,-1,1),allow_pickle=False)
    import pandas as pd
    (output/'manifests').mkdir(exist_ok=True)
    pd.DataFrame([dict(episode_id=r['episode_id'],source_recording=r['source_recording'],accepted_samples=50) for r in manifest]).to_csv(output/'manifests/episode_manifest.csv',index=False)
    ids=np.concatenate([p['episode_id'] for p in parts]);steps=np.concatenate([p['step'] for p in parts])
    assert len(set(zip(ids,steps)))==25000
    error=np.concatenate([p['next_cartesian'][:,:2]-p['reference_next_cartesian'][:,:2] for p in parts])
    save(output/'summaries/regenerated_summary.json',dict(samples=25000,episodes=500,training_recordings=43,duplicate_episode_steps=0,
         invalid_samples=0,teacher_tip_RMSE_mm=float(np.sqrt(np.mean(np.sum(error*error,axis=1)))*1000),external_Test_files_opened=0,external_Test_files_modified=0,
         scope='Regenerated measurements; accepted publication results are not overwritten. Timing and solver numerics can depend on platform.'))
    inventory={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['episodes','arrays'] for p in sorted((output/folder).iterdir()) if p.is_file()}
    save(output/'metadata/corpus_sha256.json',inventory)
    assert not engine.DENIED
    print('Teacher corpus complete:',output)

if __name__=='__main__':main()
