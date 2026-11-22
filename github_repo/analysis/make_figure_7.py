"""Figure 7 (ablation recall and identity probe)."""
import pandas as pd, numpy as np, matplotlib
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'main_10seeds')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
d=pd.read_csv(os.path.join(RES, 'all.csv')); p4=d[d.regime=='P4']; pr=pd.read_csv(os.path.join(RES, 'probe.csv'))
C=[f'abl_c{i}' for i in range(1,9)]
L=['1\nLSTM\n16AU','2\nLSTM\n44','3\nTCN','4\nMSTCN\nCE','5\nMSTCN\nFocal','6\nλ=0.1','7\nλ=0.5','8\nλ=1.0']
ink,sec,mut,grid='#0b0b0b','#52514e','#898781','#e6e5e1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':mut,'xtick.color':sec,'ytick.color':sec})
W=7.5; fig,ax=plt.subplots(1,2,figsize=(W,W/2.4142))
x=np.arange(8); w=0.38
q=pr.groupby(['model','seed'])[['probe_acc','probe_acc_naive']].mean()*100
for j,(col,lab,c) in enumerate([('probe_acc','Purged block CV','#2a78d6'),('probe_acc_naive','Naive random CV','#eb6834')]):
    mu=[q.loc[m][col].mean() for m in C]; sd=[q.loc[m][col].std() for m in C]
    ax[0].bar(x+(j-.5)*w,mu,w-0.04,yerr=sd,color=c,label=lab,error_kw=dict(lw=0.8,capsize=1.5,ecolor=sec),edgecolor='white',linewidth=0.6)
ax[0].axhline(4.55,color=mut,ls=(0,(3,2)),lw=1,label='Chance 4.55%')
ax[0].set_ylabel('Subject-ID probe accuracy (%)'); ax[0].set_title('(a) Linear probe of subject identity from z',loc='left',fontsize=8.5,color=ink)
ax[0].legend(frameon=False,fontsize=6.8,loc='upper right')
cols=['#2a78d6','#eb6834','#1baf7a']; names=['Disengaged','Engaged','Highly Engaged']; w3=0.27
R=np.array([[recall_score(g.true_label,g['pred_'+m],labels=[0,1,2],average=None,zero_division=0)*100 for _,g in p4.groupby('seed')] for m in C])
for k in range(3):
    ax[1].bar(x+(k-1)*w3,R[:,:,k].mean(1),w3-0.03,yerr=R[:,:,k].std(1,ddof=1),color=cols[k],label=names[k],error_kw=dict(lw=0.7,capsize=1.2,ecolor=sec),edgecolor='white',linewidth=0.5)
ax[1].set_ylabel('Recall at P4 (%, mean ± SD, 10 seeds)'); ax[1].set_title('(b) Class recall per configuration',loc='left',fontsize=8.5,color=ink)
ax[1].legend(frameon=False,fontsize=6.8,loc='upper left',ncol=3)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(L,fontsize=6.2); a.grid(axis='y',color=grid,lw=0.7); a.set_axisbelow(True)
    for s in ['top','right']: a.spines[s].set_visible(False)
ax[0].set_ylim(0,80); ax[1].set_ylim(0,105)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig7_ablation_probe.png'),dpi=300,facecolor='white')

