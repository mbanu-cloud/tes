"""Cek otomatis folder hasil sebelum dipakai di naskah. Gagal (exit 1) jika ada yang tidak sinkron.

  python verify_results.py results/20261123_0900 [--classical-from all_lama.csv]

--classical-from: gabungkan kolom 5 model klasik dari file OOF lama (jika run final memakai --skip-classical).
Menulis <folder>/all.csv yang siap dibaca stats_all.py.
"""
import sys, json, argparse, numpy as np, pandas as pd, yaml

ap = argparse.ArgumentParser(); ap.add_argument('folder'); ap.add_argument('--classical-from')
a = ap.parse_args(); F = a.folder.rstrip('/')
cfg = yaml.safe_load(open(f'{F}/config.yaml'))
oof = pd.read_csv(f'{F}/oof_all.csv'); folds = pd.read_csv(f'{F}/folds.csv')
errors, notes = [], []
def check(cond, msg):
    (notes if cond else errors).append(('OK   ' if cond else 'GAGAL ') + msg)

CL = ['decision_tree', 'svm_rbf', 'gradient_boosting', 'xgboost', 'random_forest']
if a.classical_from:
    old = pd.read_csv(a.classical_from)
    cols = ['seed', 'regime', 'sequence_idx'] + [c for c in old.columns if any(c.startswith(p + m) for m in CL for p in ('pred_', 'prob_'))]
    oof = oof.merge(old[cols], on=['seed', 'regime', 'sequence_idx'], how='left', suffixes=('', '_dup'))
    oof = oof[[c for c in oof.columns if not c.endswith('_dup')]]

# 0. reproduksi kolom deep lama (impl: user, 8 epoch, seed sama -> harus identik dengan file OOF lama)
if a.classical_from and cfg['deep'].get('impl') == 'user':
    old = pd.read_csv(a.classical_from)
    mm = oof.merge(old[['seed', 'regime', 'sequence_idx', 'pred_bilstm_44feat', 'pred_dasi_msnet']],
                   on=['seed', 'regime', 'sequence_idx'], suffixes=('', '_old'))
    for c in ['bilstm_44feat', 'dasi_msnet']:
        agree = (mm[f'pred_{c}'] == mm[f'pred_{c}_old']).mean() * 100
        check(agree == 100, f'{c} mereproduksi file OOF lama: {agree:.2f}% prediksi identik')

# 1. jumlah baris
n_exp = {'P1': 1044, 'P2': 1044, 'P3': 219, 'P4': 1044}
for (r, s), g in oof.groupby(['regime', 'seed']):
    check(len(g) == n_exp[r], f'{r} seed {s}: {len(g)} baris (harus {n_exp[r]})')
check(sorted(oof.seed.unique()) == cfg['seeds'], 'semua seed ada')
# 2. fold sama dengan folds.csv
m = oof.merge(folds, on=['seed', 'regime', 'sequence_idx'], suffixes=('', '_f'))
check(len(m) == len(oof) and (m.fold == m.fold_f).all(), 'fold identik dengan folds.csv')
# 3. P4 subject-disjoint
p4 = oof[oof.regime == 'P4']
check((p4.groupby(['seed', 'subject_id']).fold.nunique() == 1).all(), 'P4: tiap subjek hanya di satu fold')
# 4. tidak ada kolom kosong
pred_cols = [c for c in oof.columns if c.startswith('pred_') and not c.startswith('pred_abl_')]
for c in pred_cols:
    check(oof[c].notna().all(), f'{c} lengkap di semua regime')
for c in [c for c in oof.columns if c.startswith('pred_abl_')]:
    check(p4[c].notna().all(), f'{c} lengkap di P4')
# 5. identitas ablation vs model utama
def same(a_, b_):
    if f'pred_{a_}' not in p4 or f'pred_{b_}' not in p4: return False
    pa = p4[[f'prob_{a_}_c{k}' for k in range(3)]].values; pb = p4[[f'prob_{b_}_c{k}' for k in range(3)]].values
    return (p4[f'pred_{a_}'] == p4[f'pred_{b_}']).all() and np.allclose(pa, pb, atol=1e-5)
check(same('abl_c7', 'dasi_msnet'), 'ablation c7 identik dengan dasi_msnet')
check(same('abl_c2', 'bilstm_44feat'), 'ablation c2 identik dengan bilstm_44feat')
# 6. epoch & waktu
t = pd.read_csv(f'{F}/timing.csv')
if 'epochs_run' in t:
    e = t.dropna(subset=['epochs_run'])
    check((e.epochs_run <= cfg['deep']['max_epochs']).all(), f"epoch <= max_epochs ({cfg['deep']['max_epochs']})")
    notes.append('INFO epoch dijalankan (median per model): ' + e.groupby('model').epochs_run.median().to_dict().__str__())
notes.append('INFO waktu CPU per seed (detik, jumlah 5 fold): ' +
             t.groupby(['model', 'seed', 'regime']).cpu_s.sum().groupby('model').agg(['mean', 'std']).round(2).to_dict('index').__str__())
# 7. probe
try:
    pr = pd.read_csv(f'{F}/probe.csv')
except pd.errors.EmptyDataError:
    pr = pd.DataFrame(columns=['model', 'seed', 'probe_acc'])
check(len(pr) > 0, 'probe.csv berisi data')
deep_models = sorted(pr.model.unique())
for mdl in deep_models:
    check(len(pr[pr.model == mdl]) == len(cfg['seeds']) * 5, f'probe {mdl}: {len(pr[pr.model == mdl])} baris (harus {len(cfg["seeds"]) * 5})')
notes.append('INFO probe (mean ± SD per seed): ' + pr.groupby(['model', 'seed']).probe_acc.mean().groupby('model')
             .agg(['mean', 'std']).mul(100).round(2).to_dict('index').__str__())

print('\n'.join(notes + errors))
oof.to_csv(f'{F}/all.csv', index=False)
if errors:
    print(f'\n{len(errors)} pengecekan GAGAL. Jangan dipakai untuk naskah sebelum diperbaiki.'); sys.exit(1)
print(f'\nSemua pengecekan lolos. File siap: {F}/all.csv, probe.csv, timing.csv, train_log.csv')
