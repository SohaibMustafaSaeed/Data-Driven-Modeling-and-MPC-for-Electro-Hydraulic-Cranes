import hashlib
import json
import os
from pathlib import Path
import sys

RELEASE=Path(__file__).resolve().parents[1]
PROJECT=RELEASE.parent
sys.path.insert(0,str(PROJECT))
import teacher_engine as base

ROOT=Path(os.environ.get('NEURAL_OUTPUT_DIR',RELEASE)).expanduser().resolve()
DATA_ROOT=Path(os.environ.get('MPC_TEACHER_DATASET_DIR',RELEASE/'dataset/generated')).expanduser().resolve()
FEATURE_SCHEMA=RELEASE/'dataset/metadata/feature_schema.json'
PROTECTED=[DATA_ROOT,RELEASE/'models',RELEASE/'configs',RELEASE/'data_splits',RELEASE/'dataset/metadata',
           RELEASE/'results',RELEASE/'figures',RELEASE/'improvement_study',
           *[PROJECT/folder/'artifacts' for folder in ['GRU Latent','LSTM','Transformers','Compact Recurrent']],
           PROJECT/'MPC offline Model/artifacts',PROJECT/'MPC offline Model/configs']

def audit(event,args):
    base.audit(event,args)
    if event not in ['open','os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.utime']:return
    for value in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(value,(str,bytes,os.PathLike)):continue
        path=Path(os.fsdecode(value)).resolve()
        if not any(path.is_relative_to(folder) for folder in PROTECTED):continue
        write=event!='open'
        if event=='open':
            mode,flags=args[1:3]
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write:raise PermissionError(f'Frozen release artifact or teacher corpus: {path}')

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot():
    return {str(path):digest(path) for folder in PROTECTED for path in sorted(folder.rglob('*')) if path.is_file()}

def verify_immutable():
    base.verify_frozen_configuration()
    reference=ROOT/'configs/accepted_sha256.json'
    if reference.exists():assert snapshot()==json.loads(reference.read_text()),'Frozen input artifacts changed'
    for artifact in json.loads((RELEASE/'configs/release_checkpoint_manifest.json').read_text()).values():
        assert digest(PROJECT/artifact['checkpoint'])==artifact['sha256'],'Released checkpoint changed'
    assert not base.DENIED,'External Test access attempted'

def load_episode(episode_id):
    import numpy as np
    with np.load(DATA_ROOT/'episodes'/(str(episode_id)+'.npz'),allow_pickle=False) as arrays:return {key:arrays[key] for key in arrays.files}

def initial_rows(data,ctx):
    import numpy as np
    current=np.empty(24,float);current[ctx.core.SI]=data['full_state'][0];current[ctx.core.CI]=data['previous_command'][0]
    return np.stack((data['previous_full_input'][0],current))
