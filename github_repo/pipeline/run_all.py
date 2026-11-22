"""Satu runner untuk seluruh eksperimen audit kebocoran WACV.

Pemakaian:
  python run_all.py --epoch-test            # langkah 1: uji konvergensi Bi-LSTM & DASI di P1 (seed 42)
  python run_all.py                         # langkah 3: run final (semua regime, seed, model, ablation, probe)
  python run_all.py --skip-classical        # model klasik tidak dijalankan ulang (pakai hasil lama)

Semua fold dibaca dari folds.csv (dibekukan). Semua setelan dibaca dari config.yaml.
"""
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS', '1'); _os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse, json, os, platform, subprocess, sys, time, hashlib
import numpy as np, pandas as pd, yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='config.yaml')
ap.add_argument('--epoch-test', action='store_true')
ap.add_argument('--skip-classical', action='store_true')
ap.add_argument('--skip-deep', action='store_true')
ap.add_argument('--seeds', type=int, nargs='*', help='subset seed (untuk uji cepat)')
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
D = cfg['data']

# ---------------- data ----------------
F = np.load(D['features_file'])
X_all, y_all, subj_all, sidx_all = F['X'].astype(np.float32), F['y'].astype(int), F['subject'], F['sequence_idx']
assert X_all.shape[1:] == (D['window'], 44), X_all.shape
pos = {int(s): i for i, s in enumerate(sidx_all)}             # sequence_idx -> baris di X
folds = pd.read_csv(D['folds_file'])
# cek label & subjek di folds.csv cocok dengan features.npz
chk = folds.drop_duplicates('sequence_idx')
rows = chk.sequence_idx.map(pos).values
assert (y_all[rows] == chk.true_label.values).all(), 'label features.npz != folds.csv'
assert (subj_all[rows].astype(str) == chk.subject_id.astype(str).values).all(), 'subjek features.npz != folds.csv'

def pool(X):                                                # mean || std temporal -> 88 dim
    return np.hstack([X.mean(1), X.std(1)])

def classical_models(seed):  # seed = seed + fold, sama seperti skrip asli
    C = cfg['classical']
    m = {
        'decision_tree': DecisionTreeClassifier(random_state=seed, **C['decision_tree']),
        'svm_rbf': make_pipeline(StandardScaler(), SVC(random_state=seed, **C['svm_rbf'])),
        'gradient_boosting': GradientBoostingClassifier(random_state=seed, **C['gradient_boosting']),
        'random_forest': RandomForestClassifier(random_state=seed, n_jobs=1, **C['random_forest']),
    }
    try:
        from xgboost import XGBClassifier
        m['xgboost'] = XGBClassifier(random_state=seed, n_jobs=1, **C['xgboost'])
    except ImportError:
        print('PERINGATAN: xgboost tidak terpasang, model xgboost dilewati', file=sys.stderr)
    return m

def full_proba(model, X, classes_present):
    p = model.predict_proba(X); out = np.zeros((len(X), 3))
    out[:, classes_present] = p
    return out

def train_test(regime, seed, fold, g):
    """g = baris folds.csv untuk (regime, seed). Kembalikan index sequence_idx latih & uji (P2 dengan purge)."""
    te = g[g.fold == fold]; tr = g[g.fold != fold]
    if regime == 'P2':
        k = D['p2_purge_windows']; keep = []
        for s, gs in g.groupby('subject_id'):
            order = gs.sort_values('sequence_idx').sequence_idx.values
            rank = {q: i for i, q in enumerate(order)}
            te_s = te[te.subject_id == s].sequence_idx.map(rank)
            tr_s = tr[tr.subject_id == s]
            if len(te_s) == 0:
                keep += list(tr_s.sequence_idx); continue
            lo, hi = te_s.min(), te_s.max()
            r = tr_s.sequence_idx.map(rank)
            keep += list(tr_s.sequence_idx[(r < lo - k) | (r > hi + k)])
        tr = tr[tr.sequence_idx.isin(keep)]
    return tr.sequence_idx.values, te.sequence_idx.values

def inner_split(regime, tr_idx, seed):
    rng = np.random.default_rng(seed)
    rows = np.array([pos[i] for i in tr_idx])
    if regime == 'P4':
        subs = np.unique(subj_all[rows]); vs = rng.choice(subs, cfg['deep']['inner_val_subjects'], replace=False)
        va = np.isin(subj_all[rows], vs)
    else:
        va = np.zeros(len(rows), bool); va[rng.choice(len(rows), int(0.15 * len(rows)), replace=False)] = True
    return rows[~va], rows[va]

def probe_subject(net, embed):
    """z dari SEMUA 1.044 sequence -> ID subjek; CV blok temporal per subjek + purge (chance 1/22)."""
    Z = embed(net, X_all); k = D['p2_purge_windows']; nf = cfg['probe']['folds']
    blk = np.zeros(len(Z), int); rank = np.zeros(len(Z), int)
    for s in np.unique(subj_all):
        idx = np.where(subj_all == s)[0]; idx = idx[np.argsort(sidx_all[idx])]
        for b, ch in enumerate(np.array_split(idx, nf)): blk[ch] = b
        rank[idx] = np.arange(len(idx))
    accs = []
    for b in range(nf):
        te = blk == b; tr = ~te
        for s in np.unique(subj_all):                        # purge
            m = subj_all == s
            if (te & m).any():
                lo, hi = rank[te & m].min(), rank[te & m].max()
                tr &= ~(m & (rank >= lo - k) & (rank <= hi + k))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['probe']['C'], max_iter=cfg['probe']['max_iter']))
        clf.fit(Z[tr], subj_all[tr]); accs.append(np.mean(clf.predict(Z[te]) == subj_all[te]))
    return float(np.mean(accs))

def probe_naive(net, embed):
    """Probe versi lama: CV acak 5-fold antar sequence (window tumpang-tindih bisa bocor). Untuk pembanding saja."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    Z = embed(net, X_all)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=cfg['probe']['C'], max_iter=cfg['probe']['max_iter']))
    return float(cross_val_score(clf, Z, subj_all, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())

# ---------------- output ----------------
out_dir = os.path.join(cfg['output_dir'], time.strftime('%Y%m%d_%H%M'))
os.makedirs(out_dir, exist_ok=True)
oof_rows, timing, probes, logs = [], [], [], []

def commit_hash():
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception: return 'no-git'

info = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.version, platform=platform.platform(),
            processor=platform.processor(), commit=commit_hash(),
            config_sha256=hashlib.sha256(open(args.config, 'rb').read()).hexdigest(),
            folds_sha256=hashlib.sha256(open(D['folds_file'], 'rb').read()).hexdigest(),
            features_sha256=hashlib.sha256(open(D['features_file'], 'rb').read()).hexdigest())
try:
    import sklearn, torch; info.update(sklearn=sklearn.__version__, torch=torch.__version__)
except ImportError: pass

USER_IMPL = cfg['deep'].get('impl', 'reference') == 'user'
if not args.skip_deep or args.epoch_test:
    if USER_IMPL:
        import models_user as MD
    else:
        import models_deep as MD

SPEC_MAIN = {'bilstm_44feat': dict(arch='bilstm', features='all44', loss='ce', grl=0.0),
             'dasi_msnet': dict(arch='mstcn', features='all44', loss='focal', grl=cfg['deep']['grl_lambda'])}

def feats(X, spec):
    if spec['features'] == 'au16':
        a, b = cfg['au16_columns']; return X[:, :, a:b]
    return X

# ---------------- langkah 1: uji konvergensi ----------------
if args.epoch_test:
    dc = dict(cfg['deep']); dc['max_epochs'] = 100; dc['patience'] = 100
    g = folds[(folds.regime == 'P1') & (folds.seed == 42)]
    for name, spec in SPEC_MAIN.items():
        for fold in range(5):
            tr, te = train_test('P1', 42, fold, g)
            rtr = np.array([pos[i] for i in tr]); rte = np.array([pos[i] for i in te])
            # diagnostik saja: data uji dipakai sebagai "val" untuk melihat kurva, BUKAN untuk seleksi model
            tag = dict(model=name, regime='P1', seed=42, fold=fold)
            if USER_IMPL:
                MD.fit_predict(dc, spec, feats(X_all[rtr], spec), y_all[rtr], subj_all[rtr], feats(X_all[rte], spec),
                               42, fold, logs, tag, y_eval=y_all[rte])
            else:
                MD.fit_predict(dc, spec, feats(X_all[rtr], spec), y_all[rtr], subj_all[rtr],
                               feats(X_all[rte], spec), y_all[rte], feats(X_all[rte], spec), 42 * 100 + fold, logs, tag)
    L = pd.DataFrame(logs); L.to_csv(os.path.join(out_dir, 'epoch_test.csv'), index=False)
    col = 'test_ba' if USER_IMPL else 'val_ba'
    print(L.groupby(['model', 'epoch'])[col].mean().unstack(0).iloc[[0, 4, 7, 9, 14, 19, 29, 49, 74, 99]].round(3))
    print('Hasil di', out_dir, '-> pilih max_epochs saat kurva mendatar, tulis ke config.yaml'); sys.exit()

# ---------------- langkah 3: run final ----------------
seeds = args.seeds or cfg['seeds']
for regime in cfg['regimes']:
    for seed in seeds:
        g = folds[(folds.regime == regime) & (folds.seed == seed)]
        rec = {int(i): {} for i in g.sequence_idx}
        for fold in sorted(g.fold.unique()):
            tr, te = train_test(regime, seed, fold, g)
            rtr = np.array([pos[i] for i in tr]); rte = np.array([pos[i] for i in te])
            if regime == 'P4':
                assert not set(subj_all[rtr]) & set(subj_all[rte]), 'subjek uji bocor ke data latih'
            # --- klasik ---
            if not args.skip_classical:
                Xtr, Xte = pool(X_all[rtr]), pool(X_all[rte])
                for name, mdl in classical_models(seed + fold).items():
                    t0 = time.process_time(); mdl.fit(Xtr, y_all[rtr])
                    present = np.unique(y_all[rtr]); P = full_proba(mdl, Xte, present)
                    timing.append(dict(regime=regime, seed=seed, fold=fold, model=name, cpu_s=time.process_time() - t0))
                    for i, q in enumerate(te): rec[int(q)][name] = P[i]
            # --- deep utama + ablation ---
            if not args.skip_deep:
                specs = {k: v for k, v in SPEC_MAIN.items() if k in cfg.get('main_deep', SPEC_MAIN)}
                if regime == 'P4':
                    specs.update({f'abl_{k}': v for k, v in (cfg.get('ablation') or {}).items()})
                if not USER_IMPL:
                    itr, iva = inner_split(regime, tr, seed * 100 + fold)
                for name, spec in specs.items():
                    tag = dict(model=name, regime=regime, seed=seed, fold=fold)
                    if USER_IMPL:   # persis skrip asli: seluruh data latih, epoch tetap, tanpa early stopping
                        P, net, inf = MD.fit_predict(cfg['deep'], spec, feats(X_all[rtr], spec), y_all[rtr], subj_all[rtr],
                                                     feats(X_all[rte], spec), seed, fold, logs, tag)
                    else:
                        P, net, inf = MD.fit_predict(cfg['deep'], spec, feats(X_all[itr], spec), y_all[itr], subj_all[itr],
                                                     feats(X_all[iva], spec), y_all[iva], feats(X_all[rte], spec),
                                                     seed * 100 + fold, logs, tag)
                    timing.append({**tag, 'cpu_s': inf['cpu_s'], 'epochs_run': inf['epochs_run']})
                    for i, q in enumerate(te): rec[int(q)][name] = P[i]
                    if regime == 'P4':
                        Xf = feats(X_all, spec)
                        probes.append({**tag, 'probe_acc': probe_subject(net, lambda n, _X: MD.embed(n, Xf)),
                                       'probe_acc_naive': probe_naive(net, lambda n, _X: MD.embed(n, Xf))})
            print(f'{regime} seed {seed} fold {fold} selesai', flush=True)
        for q, d in rec.items():
            r = pos[q]; row = dict(seed=seed, regime=regime, sequence_idx=q, subject_id=subj_all[r],
                                   fold=int(g[g.sequence_idx == q].fold.iloc[0]), true_label=int(y_all[r]))
            for name, P in d.items():
                row[f'pred_{name}'] = int(np.argmax(P))
                for c in range(3): row[f'prob_{name}_c{c}'] = round(float(P[c]), 6)
            oof_rows.append(row)
        # simpan bertahap agar run panjang tidak hilang
        pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, 'oof_all.csv'), index=False)
        pd.DataFrame(timing).to_csv(os.path.join(out_dir, 'timing.csv'), index=False)
        pd.DataFrame(probes).to_csv(os.path.join(out_dir, 'probe.csv'), index=False)
        pd.DataFrame(logs).to_csv(os.path.join(out_dir, 'train_log.csv'), index=False)

info['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
json.dump(info, open(os.path.join(out_dir, 'run_info.json'), 'w'), indent=1)
os.system(f'cp {args.config} {D["folds_file"]} {out_dir}/')
print('Selesai. Kirim folder', out_dir)
