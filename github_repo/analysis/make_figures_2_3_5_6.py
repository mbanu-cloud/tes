"""Figures 2, 3, 5 and 6. Run after pipeline/stats_all.py results/main_10seeds."""
import json, numpy as np, matplotlib
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', 'main_10seeds')
FIG = os.path.join(ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

o = json.load(open(os.path.join(RES, 'results.json')))
M = ['decision_tree', 'svm_rbf', 'gradient_boosting', 'xgboost', 'bilstm_44feat', 'random_forest', 'dasi_msnet']
NAME = {'decision_tree': 'Decision Tree', 'svm_rbf': 'SVM (RBF)', 'gradient_boosting': 'Gradient Boosting',
        'xgboost': 'XGBoost', 'bilstm_44feat': 'Bi-LSTM (44 feat.)', 'random_forest': 'Random Forest',
        'dasi_msnet': 'DASI-MSNet'}
COL = dict(zip(M, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']))
MK = dict(zip(M, ['o', 's', '^', 'D', 'v', 'P', 'X']))
ink, sec, mut, grid, surf = '#0b0b0b', '#52514e', '#898781', '#e6e5e1', '#ffffff'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': mut, 'axes.labelcolor': ink,
                     'xtick.color': sec, 'ytick.color': sec, 'axes.titlesize': 8.5})
OUT = FIG + '/'
W = 7.5
R = ['P1', 'P2', 'P3', 'P4']
RL = ['P1\nrandom CV', 'P2\ntemporal blocks', 'P3\nstride 10', 'P4\nsubject-disjoint']

# ---------- Figure 2: accuracy and balanced accuracy across regimes ----------
fig, axes = plt.subplots(1, 2, figsize=(W, W / 1.8966))
for ax, met, ref, reflab, title in [(axes[0], 'acc', 56.99, 'Majority baseline 56.99%', '(a) Accuracy'),
                                    (axes[1], 'ba', 33.33, 'Chance 33.33%', '(b) Balanced accuracy')]:
    ax.axhline(ref, color=mut, lw=1, ls=(0, (3, 2)), zorder=1)
    ends = []
    for m in M:
        mu = [o['seed'][m][r][met][0] for r in R]; sd = [o['seed'][m][r][met][1] for r in R]
        ax.errorbar(range(4), mu, yerr=sd, color=COL[m], marker=MK[m], ms=4.5, lw=1.6, capsize=2,
                    elinewidth=0.9, markeredgecolor=surf, markeredgewidth=0.6, zorder=3)
        ends.append([mu[-1], m])
    # direct labels at the right, de-collided
    ends.sort()
    ys = [e[0] for e in ends]; gap = 2.6
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap: ys[i] = ys[i - 1] + gap
    for (yv, m), yl in zip(ends, ys):
        ax.text(3.15, yl, NAME[m], color=ink, fontsize=6.3, va='center')
        ax.plot([3.02, 3.13], [yv, yl], color=COL[m], lw=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(RL, fontsize=6.8)
    ax.set_xlim(-0.2, 4.15)
    ax.set_title(title + '   (dashed: ' + reflab + ')', loc='left', color=ink)
    ax.set_ylabel('Percent (mean ± SD, 10 seeds)')
    ax.grid(axis='y', color=grid, lw=0.7); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axes[0].set_ylim(45, 82); axes[1].set_ylim(25, 78)
fig.tight_layout()
fig.savefig(OUT + 'fig2_regimes.png', dpi=300, facecolor=surf)

# ---------- Figure 3: post-hoc permutation nulls vs observed ensemble BA ----------
nulls = np.load(os.path.join(RES, 'nulls.npz'))
order = sorted(M, key=lambda m: -o['ens_p4'][m]['ba'])
fig, ax = plt.subplots(figsize=(W, W / 2.3675))
for i, m in enumerate(order):
    nl = nulls[m]
    vp = ax.violinplot(nl, positions=[i], widths=0.7, showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor('#d9d8d3'); b.set_edgecolor('#b5b4ae'); b.set_alpha(1)
    q = np.percentile(nl, [2.5, 97.5])
    ax.plot([i, i], q, color=mut, lw=1)
    ob = o['ens_p4'][m]['ba']; p = o['ens_p4'][m]['perm_p']
    ax.scatter([i], [ob], s=42, color=COL[m], marker=MK[m], edgecolor=surf, linewidth=0.8, zorder=4)
    ptxt = f'p = {p:.4f}' if p < 0.001 else f'p = {p:.3f}'
    ax.text(i, max(ob, q[1]) + 1.6, f'{ob:.1f}%\n{ptxt}', ha='center', va='bottom', fontsize=6.5, color=ink)
ax.axhline(33.33, color=mut, lw=1, ls=(0, (3, 2)))
ax.text(len(M) - 0.55, 33.6, 'Chance 33.33%', fontsize=6.5, color=sec, ha='right', va='bottom')
ax.set_xticks(range(len(M))); ax.set_xticklabels([NAME[m] for m in order], fontsize=7)
ax.set_ylabel('Balanced accuracy (%)'); ax.set_ylim(22, 58); ax.set_xlim(-0.6, len(M) - 0.4)
ax.grid(axis='y', color=grid, lw=0.7); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + 'fig3_permutation.png', dpi=300, facecolor=surf)

# ---------- Figures 5 and 6: ensemble confusion matrices and ROC ----------
E = np.load(os.path.join(RES, 'ens_probs.npz')); y = E['y']
T = [('xgboost', '(a) XGBoost'), ('bilstm_44feat', '(b) Bi-LSTM, 44 features'), ('dasi_msnet', '(c) DASI-MSNet (GRL, λ = 0.5)')]
seq = LinearSegmentedColormap.from_list('b', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
fig, axes = plt.subplots(1, 3, figsize=(W, W / 2.2471))
for ax, (m, title) in zip(axes, T):
    pred = E[m].argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(cmn, cmap=seq, vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            v = cmn[i, j]; c = '#ffffff' if v > 55 else ink
            ax.text(j, i - 0.1, f'{v:.1f}%', ha='center', va='center', color=c, fontsize=7.5, fontweight='bold')
            ax.text(j, i + 0.22, f'n = {cm[i, j]}', ha='center', va='center', color=c, fontsize=6)
    ax.set_title(f'{title}\nBalanced accuracy {np.mean(np.diag(cmn)):.1f}%', loc='left', color=ink, fontsize=7.5)
    lab = ['Dis. (0)', 'Eng. (1)', 'High (2)']
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(lab, fontsize=6.5)
    ax.set_yticklabels(lab if ax is axes[0] else [], fontsize=6.5)
    ax.set_xlabel('Predicted label')
    if ax is axes[0]: ax.set_ylabel('True label')
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which='minor', color=surf, linewidth=2); ax.tick_params(which='both', length=0)
fig.tight_layout()
fig.savefig(OUT + 'fig5_confusion.png', dpi=300, facecolor=surf)

CL = ['Disengaged', 'Engaged', 'Highly Engaged']
cols = ['#2a78d6', '#eb6834', '#1baf7a']; ls = ['-', '--', ':']
fig, axes = plt.subplots(1, 3, figsize=(W, W / 2.2786), sharey=True)
for ax, (m, title) in zip(axes, T):
    P = E[m]; aucs = []
    for k in range(3):
        fpr, tpr, _ = roc_curve((y == k).astype(int), P[:, k]); a = roc_auc_score((y == k).astype(int), P[:, k]); aucs.append(a)
        ax.plot(fpr, tpr, color=cols[k], lw=1.6, ls=ls[k], drawstyle='steps-post', label=f'{CL[k]}: {a:.2f}')
    ax.plot([0, 1], [0, 1], color=mut, lw=0.8, ls=(0, (2, 2)), label='Chance: 0.50')
    ax.set_title(f'{title}\nMacro AUC {np.mean(aucs):.3f}', loc='left', color=ink, fontsize=7.5)
    ax.set_xlabel('False positive rate')
    if ax is axes[0]: ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
    ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    ax.legend(loc='lower right', fontsize=5.8, frameon=False, labelcolor=ink, title='AUC', title_fontsize=6)
fig.tight_layout()
fig.savefig(OUT + 'fig6_roc.png', dpi=300, facecolor=surf)
print('ok')
