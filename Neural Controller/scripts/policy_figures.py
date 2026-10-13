import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment import ROOT


def make_figures(imitation,predictions,parts,episodes,traces,timing,comparison):
    folder=ROOT/'figures'
    for architecture in ['mlp','gru']:
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        for seed in [42,123,2026]:
            history=pd.read_csv(ROOT/'models'/architecture/f'seed_{seed}'/'training_history.csv')
            ax.plot(history.epoch,history.train_MSE,label=f'Train seed {seed}',alpha=.65)
            ax.plot(history.epoch,history.validation_MSE,'--',label=f'Validation seed {seed}')
        ax.set(xlabel='Epoch',ylabel='Command MSE [VÂ²]',title=architecture.upper()+' imitation training',yscale='log');ax.legend(fontsize=8);ax.grid(alpha=.25)
        fig.savefig(folder/f'training_validation_loss_{architecture}.png',dpi=150);plt.close(fig)
    teacher=np.stack([p['target_command'] for p in parts])
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for row,(label,prediction) in enumerate(predictions.items()):
        for j,boom in enumerate(['Lift','Jib']):
            ax=axes[row,j];ax.hexbin(teacher[:,:,j].ravel(),prediction[:,:,j].ravel(),gridsize=45,bins='log',mincnt=1)
            ax.plot([-1,1],[-1,1],color='red');ax.set(xlabel='MPC command [V]',ylabel='Policy command [V]',title=label+' â€” '+boom)
    fig.savefig(folder/'command_teacher_vs_policy.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for j,boom in enumerate(['Lift','Jib']):
        for label,prediction in predictions.items():axes[j].hist((prediction-teacher)[:,:,j].ravel(),bins=60,density=True,histtype='step',label=label)
        axes[j].set(xlabel=boom+' command error [V]',ylabel='Density');axes[j].legend();axes[j].grid(alpha=.25)
    fig.savefig(folder/'command_error_histograms.png',dpi=150);plt.close(fig)
    # Representative selection uses median MPC tracking difficulty, not best policy performance.
    teacher_rows=episodes[(episodes.controller=='MPC')&(~episodes.failed)].copy()
    index=(teacher_rows.tip_position_RMSE_mm-teacher_rows.tip_position_RMSE_mm.median()).abs().idxmin()
    episode_id=teacher_rows.loc[index,'episode_id']
    data=next(data for data,trace in traces['MPC'] if data['episode_id'][0]==episode_id)
    selected={label:next(trace for d,trace in pairs if d['episode_id'][0]==episode_id) for label,pairs in traces.items() if any(d['episode_id'][0]==episode_id for d,trace in pairs)}
    reference=np.vstack((data['current_cartesian'][0],data['reference_next_cartesian']));t=np.arange(51)*.1
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained');ax.plot(reference[:,0],reference[:,1],color='black',linewidth=2,label='Desired reference')
    for label in ['MPC','MLP','GRU policy']:
        if label in selected:ax.plot(selected[label]['cartesian'][:,0],selected[label]['cartesian'][:,1],'--',label=label)
    ax.set(xlabel='Tip x [m]',ylabel='Tip z [m]',title='Held-out internal episode '+episode_id);ax.legend();ax.grid(alpha=.25)
    fig.savefig(folder/'closed_loop_tip_trajectory_xy_or_xz.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    for label in ['MPC','MLP','GRU policy']:
        if label in selected:ax.plot(t,np.linalg.norm(selected[label]['cartesian'][:,:2]-reference[:,:2],axis=1)*1000,label=label)
    ax.set(xlabel='Episode time [s]',ylabel='Tip-position error [mm]',title='Internal episode '+episode_id);ax.legend();ax.grid(alpha=.25)
    fig.savefig(folder/'closed_loop_tracking_error_vs_time.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    for j,boom in enumerate(['Lift','Jib']):
        for label in ['MPC','MLP','GRU policy']:
            if label in selected:axes[j].step(t,np.r_[selected[label]['commands'][:,j],selected[label]['commands'][-1,j]],where='post',label=label)
        axes[j].set(ylabel=boom+' command [V]');axes[j].legend();axes[j].grid(alpha=.25)
    axes[-1].set_xlabel('Episode time [s]');fig.suptitle('Closed-loop commands â€” '+episode_id)
    fig.savefig(folder/'command_comparison_vs_time.png',dpi=150);plt.close(fig)
    # Teacher-forced imitation commands, distinct from recursively controlled trajectories.
    i=next(i for i,part in enumerate(parts) if part['episode_id'][0]==episode_id)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
    for j,boom in enumerate(['Lift','Jib']):
        axes[j].plot(np.arange(50)*.1,teacher[i,:,j],color='black',label='MPC teacher')
        for label,prediction in predictions.items():axes[j].plot(np.arange(50)*.1,prediction[i,:,j],'--',label=label)
        axes[j].set(ylabel=boom+' command [V]');axes[j].legend();axes[j].grid(alpha=.25)
    axes[-1].set_xlabel('Episode time [s]');fig.suptitle('Teacher-forced imitation â€” '+episode_id)
    fig.savefig(folder/'representative_command_episode.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for label in ['MPC','MLP','GRU policy']:
        rows=episodes[(episodes.controller==label)&(~episodes.failed)]
        ax.hist(rows.tip_position_RMSE_mm,bins=25,histtype='step',label=label)
    ax.set(xlabel='Episode tip RMSE [mm]',ylabel='Episodes',title='Held-out internal source recordings');ax.legend();ax.grid(alpha=.25)
    fig.savefig(folder/'episode_rmse_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for architecture,label in [('mlp','MLP'),('gru','GRU policy')]:
        values=np.load(ROOT/'results/realtime'/(architecture+'_latency_ms.npy'))
        ax.hist(values,bins=80,histtype='step',label=label)
    ax.axvline(100,color='red',label='100-ms deadline');ax.set(xlabel='Complete policy-call latency [ms]',ylabel='Calls',xscale='log');ax.legend();ax.grid(alpha=.25)
    fig.savefig(folder/'latency_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    for row in comparison:
        if row['controller']=='Recorded-command replay':continue
        ax.scatter(row['mean_latency_ms'],row['tip_position_RMSE_mm'],s=90)
        ax.annotate(row['controller'],(row['mean_latency_ms'],row['tip_position_RMSE_mm']),xytext=(5,5),textcoords='offset points')
    ax.axvline(100,color='red',linestyle='--',label='Controller deadline');ax.set(xscale='log',xlabel='Mean decision latency [ms]',ylabel='Tip RMSE [mm]',title='Internal control accuracy versus implementation latency')
    ax.legend();ax.grid(alpha=.25);fig.savefig(folder/'accuracy_latency_comparison.png',dpi=150);plt.close(fig)
