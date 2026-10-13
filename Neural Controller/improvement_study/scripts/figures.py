import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_figures(root,parts,pred,pairs,frames,metrics,latencies,realtime,masks):
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'figure.dpi':130})
    def finish(fig,name):
        fig.tight_layout();fig.savefig(root/'figures'/name,dpi=220,bbox_inches='tight');plt.close(fig)
    selected=json.loads((root/'models/selection.json').read_text())
    hist=pd.read_csv(root/'models'/selected['name']/'training_history.csv')
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(hist.epoch,hist.train_loss,label='Train');ax.plot(hist.epoch,hist.validation_loss,label='Validation');ax.set(xlabel='Epoch',ylabel='Weighted objective');ax.legend();finish(fig,'training_validation_loss_improved_gru.png')
    # Fixed first internal episode: complete five seconds, not a cropped interval.
    y=parts[0]['target_command'];t=np.arange(50)*.1
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t,y[:,j],label='MPC teacher',color='black')
        for label,p in pred.items():ax.plot(t,p[0,:,j],label=label)
        ax.set(ylabel=['Lift command [V]','Jib command [V]'][j]);ax.legend()
    axes[-1].set_xlabel('Episode time [s]');finish(fig,'teacher_vs_baseline_vs_improved_commands.png')
    raw=np.stack([d['target_command'] for d in parts]);errors={l:(p-raw).reshape(-1,2) for l,p in pred.items()}
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,ax in enumerate(axes):
        for label,e in errors.items():ax.hist(e[:,j],bins=70,histtype='step',label=label)
        ax.set(xlabel=['Lift error [V]','Jib error [V]'][j],ylabel='Samples');ax.legend()
    finish(fig,'command_error_histograms_baseline_vs_improved.png')
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for j,ax in enumerate(axes):
        ax.plot(t[1:],np.diff(y[:,j]),label='Teacher',color='black')
        for label,p in pred.items():ax.plot(t[1:],np.diff(p[0,:,j]),label=label)
        ax.set_ylabel(['Lift delta [V]','Jib delta [V]'][j]);ax.legend()
    axes[-1].set_xlabel('Episode time [s]');finish(fig,'command_delta_teacher_vs_policies.png')
    fig,ax=plt.subplots(figsize=(7,6));ref=parts[0]['reference_preview'][:,0];ax.plot(ref[:,0],ref[:,1],color='black',label='Reference')
    for label,ps in pairs.items():tr=ps[0][1]['cartesian'];ax.plot(tr[:,0],tr[:,1],label=label)
    ax.set(xlabel='Tip x [m]',ylabel='Tip z [m]',title='Complete first internal-test episode');ax.axis('equal');ax.legend();finish(fig,'closed_loop_tip_trajectory_baseline_vs_improved.png')
    fig,ax=plt.subplots(figsize=(10,4))
    for label,ps in pairs.items():
        e=np.stack([np.linalg.norm(tr['cartesian'][1:,:2]-d['reference_preview'][:,0,:2],axis=1)*1000 for d,tr in ps])
        ax.plot(t+.1,e.mean(0),label=label);ax.fill_between(t+.1,np.percentile(e,10,axis=0),np.percentile(e,90,axis=0),alpha=.1)
    ax.set(xlabel='Episode time [s]',ylabel='Tip error [mm]',title='All 75 episodes: mean and 10â€“90% range');ax.legend();finish(fig,'tracking_error_vs_time_baseline_vs_improved.png')
    fig,ax=plt.subplots(figsize=(8,4));ax.boxplot([f.tip_position_RMSE_mm for f in frames.values()],tick_labels=list(frames));ax.set_ylabel('Episode tip RMSE [mm]');finish(fig,'episode_rmse_distribution_baseline_vs_improved.png')
    source=pd.DataFrame({label:f.groupby('source_recording').tip_position_RMSE_mm.apply(lambda s:np.sqrt(np.mean(s*s))) for label,f in frames.items()})
    source.to_csv(root/'results/comparison/per_source_RMSE.csv');fig,ax=plt.subplots(figsize=(10,5));source.plot.bar(ax=ax);ax.set_ylabel('Source tip RMSE [mm]');ax.set_xticklabels([f'Source {i+1}' for i in range(len(source))],rotation=0);finish(fig,'per_source_rmse_baseline_vs_improved.png')
    for maskname,filename in [('near_boundary','near_boundary_command_error.png'),('large_transition','large_transition_command_error.png')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for j,ax in enumerate(axes):
            ax.boxplot([abs(e[masks[maskname],j]) for e in errors.values()],tick_labels=list(errors),showfliers=False);ax.set_ylabel(['Lift absolute error [V]','Jib absolute error [V]'][j])
        finish(fig,filename)
    fig,ax=plt.subplots(figsize=(8,4))
    for label,lat in latencies.items():ax.hist(lat,bins=np.linspace(0,.25,100),histtype='step',label=label)
    ax.set(xlabel='Complete policy call [ms]',ylabel='Calls',title='10,000 calls; tail maxima reported numerically');ax.legend();finish(fig,'latency_distribution_baseline_vs_improved.png')
    fig,ax=plt.subplots(figsize=(8,5))
    for label,met in metrics.items():
        ax.scatter(realtime[label]['mean_ms'],met['tip_position_RMSE_mm']);ax.annotate(label,(realtime[label]['mean_ms'],met['tip_position_RMSE_mm']),xytext=(6,6),textcoords='offset points')
    ax.axvline(100,color='red',linestyle='--',label='100 ms deadline');ax.set(xscale='log',xlabel='Mean call / archived solve [ms]',ylabel='Tip position RMSE [mm]',title='MPC archived timing has a different CPU-load context');ax.legend();finish(fig,'accuracy_latency_baseline_vs_improved.png')
