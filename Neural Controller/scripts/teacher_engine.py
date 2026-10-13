import hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
STUDY = HERE/'dataset/metadata'
sys.path.insert(0,str(PROJECT))
from common.project import RAW_DATA_DIR, PREPARED_DIR
WORKSPACE = RAW_DATA_DIR
DATASET = PREPARED_DIR
sys.path.insert(0,str(PROJECT/'MPC offline Model'))
DENIED=[]

def audit(event,args):
    if event not in ['open','os.scandir','os.listdir','os.remove','os.rename','os.mkdir','os.rmdir','os.chmod','os.utime']:return
    for value in args[:2] if event=='os.rename' else args[:1]:
        if not isinstance(value,(str,bytes,os.PathLike)):continue
        p=Path(os.fsdecode(value)).resolve()
        if any(p.is_relative_to(root) for root in [RAW_DATA_DIR/'Test',DATASET/'testing',PROJECT/'data/Test']):
            DENIED.append(str(p));raise PermissionError(f'External Test prohibited: {p}')

def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def verify_frozen_configuration():
    current=json.loads((PROJECT/'MPC offline Model/configs/final_gru_mpc.json').read_text())
    frozen=json.loads((HERE/'dataset/metadata/frozen_teacher.json').read_text())
    for key in ['checkpoint_sha256','mechanical_checkpoint_sha256','geometry','command_bounds']:
        assert current[key]==frozen[key],f'Frozen teacher configuration changed: {key}'
    for key,value in current['arguments'].items():
        if key not in ['duration','full_recording','start_seconds']:
            assert frozen['arguments'][key]==value,f'Frozen teacher setting changed: {key}'

class Teacher:
    def __init__(self):
        import numpy as np
        import torch
        import offline_mpc as core
        from common.project import basename, resolve_array
        from crane_kinematics import PlanarGeometry
        self.np, self.core = np, core
        self.config_path = PROJECT / 'MPC offline Model/configs/final_gru_mpc.json'
        self.config = json.loads(self.config_path.read_text())
        self.settings = self.config['arguments']
        assert [self.settings[k] for k in ['block_steps', 'horizon', 'moves', 'maxiter']] == [10000, 2, 2, 5]
        torch.set_num_threads(self.settings['threads'])
        cp = PROJECT / self.config['checkpoint']
        pp = PROJECT / self.config['mechanical_checkpoint']
        for path, key in [(cp, 'checkpoint_sha256'), (pp, 'mechanical_checkpoint_sha256')]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == self.config[key]
        self.trainer, self.model, self.ck = core.load_surrogate(cp)
        self.weights = self.trainer.export_weights(self.model)
        self.plant = core.MechanicalPlant(pp)
        assert sum(p.numel() for p in self.model.parameters()) == 4098
        assert sum(p.numel() for p in self.plant.model.parameters()) == 1060
        self.mean, self.std = np.asarray(self.ck['scaler']['mean']), np.asarray(self.ck['scaler']['std'])
        self.dt = float(self.ck['scaler']['sampling_interval_seconds'])
        assert np.isclose(self.dt, 1e-5) and np.isclose(self.dt, self.plant.dt)
        self.scaler = json.loads((DATASET / 'scaler.json').read_text())
        for key in ['mean', 'std', 'input_columns', 'output_columns']:
            assert self.scaler[key] == self.ck['scaler'][key]
        self.geometry = PlanarGeometry(**self.config['geometry'])
        self.low, self.high = np.asarray(self.config['command_bounds']['low']), np.asarray(self.config['command_bounds']['high'])
        self.center, self.half = (self.low + self.high)/2, (self.high-self.low)/2
        self.neutral = np.clip(self.settings['neutral_command'], self.low, self.high)
        self.rollout = core.make_blocked_rollout(self.trainer)
        self.plant_mapping = [core.INPUT_COLS.index(col) for col in core.PLANT_INPUT_COLS]
        self.data_indices = np.r_[core.MECH_INPUT, core.CI]
        prepared = json.loads((DATASET / 'training_manifest.json').read_text())
        full = {basename(e['source_csv']): e for e in self.ck['training_config']['training_manifest']}
        mech = {basename(e['source_csv']): e for e in self.plant.checkpoint['training_config']['training_manifest']}
        self.records = {}
        for entry in prepared:
            name = basename(entry['source_csv'])
            assert name in full and name in mech
            path = resolve_array(entry, DATASET).resolve()
            assert path.parent == (DATASET / 'training').resolve()
            assert (WORKSPACE / 'Train' / name).is_file()
            count = min(int(e[key]) for e in [entry, full[name], mech[name]] for key in ['train_rows', 'validation_start'])
            self.records[name] = dict(source_recording=name, prepared_array=str(path), training_end_exclusive=count)

    def data(self, record):
        array = self.np.load(record['prepared_array'], mmap_mode='r')
        assert array.dtype == self.np.float32 and array.shape[1] == 24
        return array[:record['training_end_exclusive']]

    def physical_coverage(self, rows):
        signals = self.np.asarray(rows[:, self.data_indices], float)*self.std[self.data_indices]+self.mean[self.data_indices]
        values = self.np.column_stack((signals[:, :4], self.geometry.end_effector(signals[:, :4]), signals[:, 4:]))
        assert self.np.isfinite(values).all()
        return values

def command_bins(values):
    import numpy as np
    x = np.clip(values, -1, 1)
    result = np.full(x.shape, -1, dtype=np.int64)
    masks = [x <= -.8, (x > -.8) & (x <= -.4), (x > -.4) & (x <= -.1),
             (x > -.1) & (x < .1), (x >= .1) & (x < .4), (x >= .4) & (x < .8), x >= .8]
    for i, mask in enumerate(masks): result[mask] = i
    assert (result >= 0).all()
    return result

def episode_arrays(ctx,record):
    import numpy as np
    from scipy.optimize import minimize
    core,settings=ctx.core,ctx.settings
    x=ctx.data(record)
    start=int(record['first_retained_row'])
    assert 1<=start and start+520000<len(x)
    initial=np.asarray(x[start-1:start+1],float)*ctx.std+ctx.mean
    assert np.isfinite(initial).all()
    ctx.plant.reset(initial[:,ctx.plant_mapping])
    snapshot=ctx.plant.snapshot()
    reference=[ctx.plant.state()]
    for k in range(52):
        command=np.asarray(x[start+k*10000,core.CI],float)*ctx.std[core.CI]+ctx.mean[core.CI]
        assert np.all(command>=ctx.low-1e-6) and np.all(command<=ctx.high+1e-6)
        reference.append(ctx.plant.advance(command,internal_steps=10000))
    reference=np.asarray(reference)
    tips=ctx.geometry.end_effector(reference)
    ctx.plant.restore(snapshot)
    previous,current=np.array(x[start-1],copy=True),np.array(x[start],copy=True)
    memory=ctx.trainer.native_encode(current,ctx.weights[0],ctx.weights[1])
    previous_command=current[core.CI].astype(float)*ctx.std[core.CI]+ctx.mean[core.CI]
    seed=np.tile(np.clip((previous_command-ctx.center)/ctx.half,-1,1),(2,1)).ravel()
    ctx.rollout(previous,current,memory,np.tile(previous_command,(2,1)),10000,ctx.mean,ctx.std,ctx.weights)
    cost_weights=np.asarray([settings[k] for k in ['angle_weight','velocity_weight','effort_weight','smoothness_weight']])
    cartesian_weights=np.asarray([settings[k] for k in ['position_weight','cartesian_velocity_weight','terminal_position_weight','terminal_cartesian_velocity_weight']])
    rows=[]
    for k in range(50):
        physical=current.astype(float)*ctx.std+ctx.mean
        prior=previous.astype(float)*ctx.std+ctx.mean
        mech=ctx.plant.state()
        tip=ctx.geometry.end_effector(mech)
        memory_before=memory.copy()
        goal=(reference[k+1:k+3]-ctx.mean[core.MECH_INPUT])/ctx.std[core.MECH_INPUT]
        preview=tips[k+1:k+3].copy()
        obj=core.MPCObjective(ctx.rollout,previous,current,memory,goal,previous_command,ctx.center,ctx.half,ctx.neutral,
                             np.arange(2,dtype=np.int64),10000,ctx.mean,ctx.std,ctx.weights,cost_weights,
                             settings['terminal_angle_weight'],settings['terminal_velocity_weight'],settings['tracking_scales'],
                             ctx.geometry,preview,cartesian_weights,[settings['position_scale']]*2+[settings['cartesian_velocity_scale']]*2)
        begin=time.perf_counter()
        hold=np.tile(np.clip((previous_command-ctx.center)/ctx.half,-1,1),(2,1)).ravel()
        hold_cost,seed_cost=obj(hold),obj(seed)
        result=minimize(obj,seed,method='SLSQP',bounds=[(-1.,1.)]*4,
                        options=dict(maxiter=5,ftol=settings['ftol'],eps=settings['fd_step'],disp=False))
        obj(np.clip(result.x,-1,1))
        if obj.best_decision is None: raise RuntimeError(f'No valid action in {record["episode_id"]}, step {k}')
        plan=core.expand_moves(obj.best_decision,np.arange(2,dtype=np.int64),ctx.center,ctx.half)
        forecast,_,_,_=ctx.rollout(previous,current,memory,plan,10000,ctx.mean,ctx.std,ctx.weights)
        solve_seconds=time.perf_counter()-begin
        assert np.isfinite(plan).all() and np.all(plan>=ctx.low-1e-9) and np.all(plan<=ctx.high+1e-9)
        begin=time.perf_counter()
        previous,current,memory,predicted,next_mech,prior_full=core.advance_and_correct(
            ctx.rollout,previous,current,memory,plan[0],10000,ctx.mean,ctx.std,ctx.weights,ctx.plant)
        commit_seconds=time.perf_counter()-begin
        row=dict(episode_id=record['episode_id'],source_recording=record['source_recording'],step=k,time_s=k*.1,
                 csv_time_s=record['csv_start_s']+k*.1,retained_row=start+k*10000,source_partition='training_prefix',
                 solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),
                 iterations=int(result.nit),candidate_calls=obj.calls,invalid_candidates=obj.invalid_calls,
                 solve_seconds=solve_seconds,commit_seconds=commit_seconds,selected_cost=obj.best_cost,hold_cost=hold_cost,seed_cost=seed_cost,
                 reaching_maxiter=bool(result.nit>=5),saturated_lift=bool(np.min(abs(plan[0,0]-np.array([ctx.low[0],ctx.high[0]])))<=1e-6),
                 saturated_jib=bool(np.min(abs(plan[0,1]-np.array([ctx.low[1],ctx.high[1]])))<=1e-6),
                 full_state=physical[core.SI],previous_command=previous_command.copy(),reference_preview=preview,target_command=plan[0].copy(),
                 second_move=plan[1].copy(),current_mechanical=mech,next_mechanical=next_mech,current_cartesian=tip,
                 next_cartesian=ctx.geometry.end_effector(next_mech),reference_next_cartesian=preview[0],reference_next_mechanical=reference[k+1].copy(),
                 predicted_next_mechanical=predicted,previous_full_input=prior,predictor_memory=memory_before,
                 next_full_state=(current.astype(float)*ctx.std+ctx.mean)[core.SI])
        assert all(np.isfinite(v).all() for v in row.values() if isinstance(v,np.ndarray))
        rows.append(row)
        seed=np.clip((np.vstack((plan[1:],plan[-1]))-ctx.center)/ctx.half,-1,1).ravel()
        previous_command=plan[0].copy()
        if (k+1)%25==0: print(f'{record["episode_id"]} {record["source_recording"]}: {k+1}/50; solve={solve_seconds*1000:.1f}ms; success={result.success}',flush=True)
    arrays={key:np.stack([row[key] for row in rows]) if isinstance(rows[0][key],np.ndarray) else np.asarray([row[key] for row in rows]) for key in rows[0]}
    arrays['supervised_inputs']=np.concatenate((arrays['full_state'],arrays['previous_command'],arrays['reference_preview'].reshape(50,8)),axis=1)
    return arrays

def validate_arrays(arrays,ctx):
    import numpy as np
    n=len(arrays['step'])
    assert arrays['supervised_inputs'].shape==(n,32) and arrays['target_command'].shape==(n,2)
    assert np.array_equal(arrays['supervised_inputs'][:,:22],arrays['full_state'])
    assert np.array_equal(arrays['supervised_inputs'][:,22:24],arrays['previous_command'])
    assert np.array_equal(arrays['supervised_inputs'][:,24:],arrays['reference_preview'].reshape(n,8))
    for key,value in arrays.items():
        if value.dtype.kind in 'fi': assert np.isfinite(value).all(),key
    for key in ['target_command','second_move']:
        assert np.all(arrays[key]>=ctx.low-1e-9) and np.all(arrays[key]<=ctx.high+1e-9)
    for episode in np.unique(arrays['episode_id']):
        ids=np.flatnonzero(arrays['episode_id']==episode)
        assert len(ids)==50 and np.array_equal(arrays['step'][ids],np.arange(50))
        assert np.array_equal(arrays['previous_command'][ids[1:]],arrays['target_command'][ids[:-1]])
        assert np.array_equal(arrays['current_mechanical'][ids[1:]],arrays['next_mechanical'][ids[:-1]])
        assert np.array_equal(arrays['full_state'][ids[1:]],arrays['next_full_state'][ids[:-1]])
