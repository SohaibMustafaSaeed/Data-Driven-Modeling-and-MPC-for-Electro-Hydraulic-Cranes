import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environment import ROOT,DATA_ROOT,FEATURE_SCHEMA,save
from policy_runtime import make_model


def train_all():
    settings=json.loads((ROOT/'configs/training.json').read_text())
    X=np.load(DATA_ROOT/'arrays/X_physical.npy',mmap_mode='r')
    Y=np.load(ROOT/'data_splits/Y_command_nominal.npy',mmap_mode='r')
    scaler=json.loads((ROOT/'configs/input_scaler.json').read_text())
    mean,std=np.asarray(scaler['mean']),np.asarray(scaler['effective_std'])
    train_ids=np.load(ROOT/'data_splits/train_indices.npy');val_ids=np.load(ROOT/'data_splits/validation_indices.npy')
    train_X=torch.from_numpy(((np.asarray(X[train_ids])-mean)/std).astype(np.float32))
    val_X=torch.from_numpy(((np.asarray(X[val_ids])-mean)/std).astype(np.float32))
    train_Y=torch.from_numpy(np.asarray(Y[train_ids],np.float32));val_Y=torch.from_numpy(np.asarray(Y[val_ids],np.float32))
    input_order=[f['name'] for f in json.loads((FEATURE_SCHEMA).read_text())['supervised_inputs']]
    summaries=[]
    for architecture in ['mlp','gru']:
        for seed in settings['seeds']:
            folder=ROOT/'models'/architecture/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            history_path=folder/'training_history.csv'
            if (folder/'complete.json').exists():summaries.append(json.loads((folder/'complete.json').read_text()));continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=make_model(architecture)
            optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=10,min_lr=1e-5)
            tx=train_X if architecture=='mlp' else train_X.reshape(-1,50,32)
            ty=train_Y if architecture=='mlp' else train_Y.reshape(-1,50,2)
            vx=val_X if architecture=='mlp' else val_X.reshape(-1,50,32)
            vy=val_Y if architecture=='mlp' else val_Y.reshape(-1,50,2)
            batch=settings['mlp_batch_samples'] if architecture=='mlp' else settings['gru_batch_episodes']
            best=float('inf');best_epoch=0;stale=0;history=[];started=time.perf_counter()
            for epoch in range(1,settings['max_epochs']+1):
                model.train();order=torch.randperm(len(tx));total=0.;count=0
                for start in range(0,len(tx),batch):
                    ids=order[start:start+batch];optimizer.zero_grad(set_to_none=True)
                    prediction,_=model(tx[ids],None);loss=torch.mean((prediction-ty[ids])**2)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    total+=float(loss.detach())*len(ids);count+=len(ids)
                model.eval()
                with torch.inference_mode():
                    prediction,_=model(vx,None);validation=float(torch.mean((prediction-vy)**2))
                scheduler.step(validation)
                history.append(dict(epoch=epoch,train_MSE=total/count,validation_MSE=validation,learning_rate=optimizer.param_groups[0]['lr']))
                if validation<best-1e-7:
                    best=validation;best_epoch=epoch;stale=0
                    bundle=dict(format='neural_crane_policy_v1',architecture=architecture,weights=model.state_dict(),
                                scaler=scaler,feature_order=input_order,parameter_count=sum(p.numel() for p in model.parameters()),
                                seed=seed,best_epoch=epoch,validation_MSE=best,
                                architecture_config=dict(input=32,hidden=[64,64,32] if architecture=='mlp' else [32,16],output=2,
                                                         activation='ReLU',output_activation='tanh',stateful=architecture=='gru'),
                                output_bounds=[-1.,1.],teacher_memory_is_input=False,target_view='Documented nominal-bound projection of actual teacher commands')
                    torch.save(bundle,folder/'best_policy.pt')
                else:stale+=1
                if epoch%20==0 or stale>=settings['patience']:
                    print(f'{architecture.upper()} seed={seed} epoch={epoch}: train_MSE={total/count:.6f}; validation_RMSE={np.sqrt(validation):.5f} V; best_epoch={best_epoch}; elapsed={time.perf_counter()-started:.1f}s',flush=True)
                    pd.DataFrame(history).to_csv(history_path,index=False)
                if stale>=settings['patience']:break
            pd.DataFrame(history).to_csv(history_path,index=False)
            summary=dict(architecture=architecture,seed=seed,best_epoch=best_epoch,validation_MSE=best,validation_RMSE=float(np.sqrt(best)),
                         parameter_count=sum(p.numel() for p in model.parameters()),epochs_completed=epoch,training_seconds=time.perf_counter()-started,
                         checkpoint=str(folder/'best_policy.pt'))
            save(folder/'complete.json',summary);summaries.append(summary)
        group=[r for r in summaries if r['architecture']==architecture];chosen=min(group,key=lambda r:(r['validation_MSE'],r['seed']))
        destination=ROOT/'models'/architecture/'policy.pt';destination.write_bytes(Path(chosen['checkpoint']).read_bytes())
        save(ROOT/'models'/architecture/'selection.json',dict(**chosen,selection_basis='Lowest validation command MSE only; internal test not used.',frozen=True))
    pd.DataFrame(summaries).to_csv(ROOT/'results/comparison/training_seed_summary.csv',index=False)
    save(ROOT/'results/comparison/training_seed_summary.json',summaries)
    return summaries
