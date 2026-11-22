"""Within-subject vs pooled AUC (Section 4.4). Run after pipeline/stats_all.py."""
import numpy as np, pandas as pd, json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'main_10seeds')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
d=pd.read_csv(os.path.join(RES, 'all.csv')); p4=d[d.regime=='P4']
base=p4[p4.seed==42].sort_values('sequence_idx').reset_index(drop=True)
E=np.load(os.path.join(RES, 'ens_probs.npz')); y=E['y']; sid=base.subject_id.values
M=['decision_tree','svm_rbf','gradient_boosting','xgboost','bilstm_44feat','random_forest','dasi_msnet']
res={}
# oracle for c0
for k in [0,2]:
    sh=pd.Series(y==k).groupby(sid).transform('mean').values
    print('oracle c%d AUC'%k, round(roc_auc_score((y==k).astype(int),sh),3))
for m in M:
    P=E[m]; out={}
    for k,(a,b) in {'c2_vs_c1':(2,1),'c1_vs_c2':(1,2)}.items():
        pass
    # within-subject: Highly Engaged vs Engaged, score = P2/(P1+P2)
    aucs=[];ns=[]
    for s in np.unique(sid):
        i=(sid==s)&np.isin(y,[1,2])
        yy=y[i]
        if (yy==1).sum()>=5 and (yy==2).sum()>=5:
            sc=(P[i,2]-P[i,1]); aucs.append(roc_auc_score((yy==2).astype(int),sc)); ns.append(i.sum())
    aucs=np.array(aucs); w=wilcoxon(aucs-0.5)
    # between-subject pooled AUC for same task
    i=np.isin(y,[1,2]); pooled=roc_auc_score((y[i]==2).astype(int),(P[i,2]-P[i,1]))
    res[m]=dict(n_subj=len(aucs),within_median=round(float(np.median(aucs)),3),within_wmean=round(float(np.average(aucs,weights=ns)),3),p=round(float(w.pvalue),4),pooled=round(pooled,3), n_above=int((aucs>0.5).sum()))
    print(m,res[m])
json.dump(res,open(os.path.join(RES, 'within_subject_auc.json'),'w'),indent=1)
