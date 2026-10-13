import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(32,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2),nn.Tanh())

    def forward(self,x,hidden=None):
        return self.network(x),None


class GRUPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory=nn.GRU(32,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2),nn.Tanh())

    def forward(self,x,hidden=None):
        sequence,hidden=self.memory(x,hidden)
        return self.head(sequence),hidden


def make_model(architecture):
    return MLPPolicy() if architecture=='mlp' else GRUPolicy()


class Policy:
    def __init__(self,checkpoint):
        self.path=Path(checkpoint)
        self.bundle=torch.load(self.path,map_location='cpu',weights_only=True)
        self.architecture=self.bundle['architecture']
        self.model=make_model(self.architecture)
        self.model.load_state_dict(self.bundle['weights']);self.model.eval()
        self.mean=np.asarray(self.bundle['scaler']['mean'],np.float64)
        self.std=np.asarray(self.bundle['scaler']['effective_std'],np.float64)
        self.reset()

    def reset(self):self.hidden=None

    @torch.inference_mode()
    def step(self,physical_inputs):
        inputs=np.asarray(physical_inputs,dtype=np.float64)
        if inputs.shape!=(32,) or not np.isfinite(inputs).all():raise ValueError('32 finite physical inputs required')
        normalized=((inputs-self.mean)/self.std).astype(np.float32)
        tensor=torch.from_numpy(normalized).reshape(1,1,32) if self.architecture=='gru' else torch.from_numpy(normalized).reshape(1,32)
        prediction,self.hidden=self.model(tensor,self.hidden)
        command=prediction.reshape(2).numpy().astype(np.float64)
        command=np.clip(command,-1.,1.)
        if not np.isfinite(command).all():raise RuntimeError('Policy returned nonfinite command')
        return command

    def step_components(self,full_state,previous_command,reference_preview):
        state=np.asarray(full_state,dtype=np.float64)
        command=np.asarray(previous_command,dtype=np.float64)
        reference=np.asarray(reference_preview,dtype=np.float64)
        if state.shape!=(22,) or command.shape!=(2,) or reference.shape!=(2,4):raise ValueError('Expected 22 states, two previous commands, and 2x4 reference preview')
        return self.step(np.concatenate((state,command,reference.reshape(8))))

    @torch.inference_mode()
    def predict_episodes(self,physical_inputs):
        inputs=np.asarray(physical_inputs,np.float64)
        if inputs.ndim!=3 or inputs.shape[-1]!=32:raise ValueError('Episode batch must be BxTx32')
        normalized=((inputs-self.mean)/self.std).astype(np.float32)
        x=torch.from_numpy(normalized)
        if self.architecture=='mlp':prediction,_=self.model(x.reshape(-1,32));prediction=prediction.reshape(*x.shape[:2],2)
        else:prediction,_=self.model(x,None)
        return np.clip(prediction.numpy().astype(np.float64),-1.,1.)
