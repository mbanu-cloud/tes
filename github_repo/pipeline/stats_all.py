import pandas as pd, numpy as np, json, sys
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             f1_score, recall_score, confusion_matrix)
from scipy.stats import wilcoxon, spearmanr

d = pd.read_csv(sys.argv[1] + '/all.csv')
M = ['decision_tree', 'svm_rbf', 'gradient_boosting', 'xgboost', 'bilstm_44feat', 'random_forest', 'dasi_msnet']
R = ['P1', 'P2', 'P3', 'P4']
SEEDS = sorted(d.seed.unique())
rng = np.random.default_rng(2026)
out = {}

def probs(g, m):
    P = g[[f'prob_{m}_c{k}' for k in range(3)]].values.astype(float)
    P = np.clip(P, 1e-9, None)
    return P / P.sum(1, keepdims=True)

# 1. per-seed metrics
rows = []
for (r, s), g in d.groupby(['regime', 'seed']):
    y = g.true_label.values
    for m in M:
        p = g[f'pred_{m}'].values
        rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
        rows.append(dict(regime=r, seed=s, model=m, acc=accuracy_score(y, p) * 100,
                         ba=balanced_accuracy_score(y, p) * 100,
                         f1=f1_score(y, p, average='macro', zero_division=0) * 100,
                         r0=rec[0] * 100, r1=rec[1] * 100, r2=rec[2] * 100,
                         auc=roc_auc_score(y, probs(g, m), multi_class='ovr')))
S = pd.DataFrame(rows)
S.to_csv(sys.argv[1] + '/seed_metrics_full.csv', index=False)
agg = S.groupby(['model', 'regime']).agg(['mean', 'std'])
out['seed'] = {m: {r: {k: [round(agg.loc[(m, r), (k, 'mean')], 4), round(agg.loc[(m, r), (k, 'std')], 4)]
                       for k in ['acc', 'ba', 'f1', 'r0', 'r1', 'r2', 'auc']} for r in R} for m in M}

# 2. per-subject confusion counts per seed (for bootstrap)
subs = sorted(d.subject_id.unique())
def conf_counts(r, m):
    # array [seed, subject, 3, 3]
    C = np.zeros((len(SEEDS), len(subs), 3, 3))
    g = d[d.regime == r]
    for i, s in enumerate(SEEDS):
        gs = g[g.seed == s]
        for j, sb in enumerate(subs):
            q = gs[gs.subject_id == sb]
            if len(q):
                C[i, j] = confusion_matrix(q.true_label, q[f'pred_{m}'], labels=[0, 1, 2])
    return C
def acc_ba(Csum):  # Csum [seed,3,3]
    acc = np.trace(Csum, axis1=1, axis2=2) / Csum.sum((1, 2)) * 100
    rowt = Csum.sum(2)
    with np.errstate(invalid='ignore', divide='ignore'):
        rec = np.diagonal(Csum, axis1=1, axis2=2) / rowt
    ba = np.nanmean(rec, 1) * 100
    return acc.mean(), ba.mean()

B = 2000
picks = [rng.choice(len(subs), len(subs), replace=True) for _ in range(B)]
dec = {}
CC = {}
for m in M:
    C = {r: conf_counts(r, m) for r in ['P1', 'P2', 'P4']}
    CC[m] = C
    pt = {r: acc_ba(C[r].sum(1)) for r in C}
    bs = []
    for pk in picks:
        v = {r: acc_ba(C[r][:, pk].sum(1)) for r in C}
        bs.append([v['P1'][0] - v['P2'][0], v['P2'][0] - v['P4'][0], v['P1'][0] - v['P4'][0],
                   v['P1'][1] - v['P2'][1], v['P2'][1] - v['P4'][1], v['P1'][1] - v['P4'][1]])
    bs = np.array(bs)
    lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    names = ['acc_ov', 'acc_id', 'acc_tot', 'ba_ov', 'ba_id', 'ba_tot']
    est = [pt['P1'][0] - pt['P2'][0], pt['P2'][0] - pt['P4'][0], pt['P1'][0] - pt['P4'][0],
           pt['P1'][1] - pt['P2'][1], pt['P2'][1] - pt['P4'][1], pt['P1'][1] - pt['P4'][1]]
    dec[m] = {n: [round(e, 2), round(l, 2), round(h, 2)] for n, e, l, h in zip(names, est, lo, hi)}
    # share of overlap in total (acc), with CI
    sh = bs[:, 0] / bs[:, 2]
    dec[m]['acc_share_ov'] = [round(est[0] / est[2], 3), round(np.percentile(sh, 2.5), 3), round(np.percentile(sh, 97.5), 3)]
out['decomp'] = dec

# pooled across the 5 classical models
cls = ['decision_tree', 'svm_rbf', 'gradient_boosting', 'xgboost', 'random_forest']
bsP = []
for pk in picks:
    a = []
    for m in cls:
        v = {r: acc_ba(CC[m][r][:, pk].sum(1)) for r in ['P1', 'P2', 'P4']}
        a.append([v['P1'][0] - v['P2'][0], v['P2'][0] - v['P4'][0], v['P1'][1] - v['P2'][1], v['P2'][1] - v['P4'][1]])
    bsP.append(np.mean(a, 0))
bsP = np.array(bsP)
estP = np.mean([[dec[m]['acc_ov'][0], dec[m]['acc_id'][0], dec[m]['ba_ov'][0], dec[m]['ba_id'][0]] for m in cls], 0)
diff = bsP[:, 1] - bsP[:, 0]
diffb = bsP[:, 3] - bsP[:, 2]
out['decomp_classical_pooled'] = dict(
    acc_ov=[round(estP[0], 2)] + list(np.round(np.percentile(bsP[:, 0], [2.5, 97.5]), 2)),
    acc_id=[round(estP[1], 2)] + list(np.round(np.percentile(bsP[:, 1], [2.5, 97.5]), 2)),
    acc_id_minus_ov=[round(estP[1] - estP[0], 2)] + list(np.round(np.percentile(diff, [2.5, 97.5]), 2)),
    ba_ov=[round(estP[2], 2)] + list(np.round(np.percentile(bsP[:, 2], [2.5, 97.5]), 2)),
    ba_id=[round(estP[3], 2)] + list(np.round(np.percentile(bsP[:, 3], [2.5, 97.5]), 2)),
    ba_id_minus_ov=[round(estP[3] - estP[2], 2)] + list(np.round(np.percentile(diffb, [2.5, 97.5]), 2)))

# 3. Wilcoxon per subject (seed-averaged per-subject accuracy)
def subj_acc(r, m):
    g = d[d.regime == r]
    return g.assign(c=(g.true_label == g[f'pred_{m}'])).groupby('subject_id').c.mean().reindex(subs) * 100
wil = {}
for m in M:
    a1, a2, a4 = subj_acc('P1', m), subj_acc('P2', m), subj_acc('P4', m)
    res = {}
    for nm, x, y in [('P1_P4', a1, a4), ('P1_P2', a1, a2), ('P2_P4', a2, a4)]:
        w = wilcoxon(x, y, method='auto')
        res[nm] = dict(n_decrease=int((x > y).sum()), W=float(w.statistic), p=float(w.pvalue),
                       median=round(float(np.median(x - y)), 2))
    wil[m] = res
out['wilcoxon'] = wil

# 4. P4 ensemble
p4 = d[d.regime == 'P4']
ens = {}
base = p4[p4.seed == SEEDS[0]].sort_values('sequence_idx')[['sequence_idx', 'subject_id', 'true_label', 'fold']].reset_index(drop=True)
Pm = {}
for m in M:
    mp = p4.groupby('sequence_idx')[[f'prob_{m}_c{k}' for k in range(3)]].mean().sort_index().values
    mp = mp / mp.sum(1, keepdims=True)
    Pm[m] = mp
y = base.true_label.values
sid = base.subject_id.values
idx_by = {s: np.where(sid == s)[0] for s in subs}
nulls = {}
for m in M:
    P = Pm[m]; pred = P.argmax(1)
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None) * 100
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    pcauc = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(3)]
    bas, aucs = [], []
    for pk in picks:
        ii = np.concatenate([idx_by[subs[j]] for j in pk])
        if len(np.unique(y[ii])) < 3: continue
        bas.append(balanced_accuracy_score(y[ii], pred[ii]) * 100)
        aucs.append(roc_auc_score(y[ii], P[ii], multi_class='ovr'))
    obs = balanced_accuracy_score(y, pred) * 100
    nl = []
    for _ in range(B):
        yp = y.copy(); donors = rng.permutation(subs)
        for s, dn in zip(subs, donors):
            yp[idx_by[s]] = np.resize(y[idx_by[dn]], len(idx_by[s]))
        nl.append(balanced_accuracy_score(yp, pred) * 100)
    nl = np.array(nl); nulls[m] = nl
    ens[m] = dict(acc=round(accuracy_score(y, pred) * 100, 2), ba=round(obs, 2),
                  f1=round(f1_score(y, pred, average='macro') * 100, 2),
                  r=[round(v, 2) for v in rec], cm=cm.tolist(),
                  auc=round(roc_auc_score(y, P, multi_class='ovr'), 4), pcauc=[round(v, 4) for v in pcauc],
                  ba_ci=[round(np.percentile(bas, 2.5), 2), round(np.percentile(bas, 97.5), 2)],
                  auc_ci=[round(np.percentile(aucs, 2.5), 3), round(np.percentile(aucs, 97.5), 3)],
                  null_mean=round(nl.mean(), 2), null_sd=round(nl.std(), 2),
                  null_95=round(np.percentile(nl, 95), 2),
                  perm_p=round((np.sum(nl >= obs) + 1) / (B + 1), 4))
out['ens_p4'] = ens
np.savez(sys.argv[1] + '/nulls.npz', **nulls)
np.savez(sys.argv[1] + '/ens_probs.npz', y=y, **Pm)

# 5. majority baseline (fold-matched)
pred = np.zeros(len(base), int)
for f in base.fold.unique():
    pred[(base.fold == f).values] = base[base.fold != f].true_label.mode()[0]
out['majority'] = dict(acc=round(accuracy_score(y, pred) * 100, 2), ba=round(balanced_accuracy_score(y, pred) * 100, 2))

# 6. per-subject analysis at P4 ensemble
eng_share = base.groupby('subject_id').true_label.apply(lambda s: (s == 1).mean()).reindex(subs)
he_share = base.groupby('subject_id').true_label.apply(lambda s: (s == 2).mean())
ps = {}
for m in M:
    pred = Pm[m].argmax(1)
    acc_s = pd.Series(pred == y).groupby(sid).mean().reindex(subs) * 100
    r, p = spearmanr(acc_s, eng_share)
    ps[m] = dict(rho=round(r, 3), p=float(p))
out['subject_rho'] = ps
oracle = base.subject_id.map(he_share).values
out['oracle_c2_auc'] = round(roc_auc_score((y == 2).astype(int), oracle), 3)

# 7. between-model per-subject Wilcoxon (P4 ensemble accuracy)
def sacc(m):
    return pd.Series(Pm[m].argmax(1) == y).groupby(sid).mean().reindex(subs) * 100
bm = {}
for a_, b_ in [('xgboost', 'bilstm_44feat'), ('xgboost', 'dasi_msnet'), ('bilstm_44feat', 'dasi_msnet'),
               ('gradient_boosting', 'bilstm_44feat'), ('gradient_boosting', 'dasi_msnet')]:
    x, z = sacc(a_), sacc(b_)
    w = wilcoxon(x, z)
    bm[f'{a_}_vs_{b_}'] = dict(median=round(float(np.median(x - z)), 2), p=round(float(w.pvalue), 4))
out['between_models'] = bm

json.dump(out, open(sys.argv[1] + '/results.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ['decomp_classical_pooled', 'majority', 'oracle_c2_auc', 'between_models', 'subject_rho']}, indent=1, default=float))
