import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, time, json, shutil
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from dasi_msnet_architecture import Standard_BiLSTM, DASI_MSNet, CostSensitiveOrdinalFocalLoss

# Load dataset
REPO_ROOT = os.path.dirname(BASE_DIR)
data_path = os.path.join(REPO_ROOT, "data", "wacv_3level_features.npz")
data = np.load(data_path)
X_seq = data['X_seq']  # (1044, 10, 44)
y_seq = data['y_seq']  # (1044,)
s_seq = data['s_seq']  # (1044,)
N_total = len(y_seq)

LABEL_NAMES = {0: 'Disengaged', 1: 'Engaged', 2: 'Highly Engaged'}

# Feature pooling for tabular / classical models
X_mean = np.mean(X_seq, axis=1)
X_std = np.std(X_seq, axis=1)
X_pool = np.hstack([X_mean, X_std])  # (1044, 88)

# P3 non-overlapping sequence indices (stride = 10 frames = every 5th window)
p3_indices = []
for s in np.unique(s_seq):
    idx = np.where(s_seq == s)[0]
    p3_indices.extend(idx[::5])
p3_indices = np.array(p3_indices)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
REGIMES = ['P1', 'P2', 'P3', 'P4']

print(f"==========================================================================")
print(f"EXTRACTING OUT-OF-FOLD (OOF) PREDICTIONS FOR 7 MODELS × 4 REGIMES × 10 SEEDS")
print(f"Total Sequences: {N_total} (P3 Subsample: {len(p3_indices)} non-overlapping)")
print(f"Seeds: {SEEDS}")
print(f"==========================================================================\n")

regime_dfs = {}

for regime in REGIMES:
    reg_start = time.time()
    print(f"\n>>> PROCESSING REGIME: {regime} <<<")
    
    # Define dataset slice for this regime
    if regime == 'P3':
        cur_X_seq = X_seq[p3_indices]
        cur_X_pool = X_pool[p3_indices]
        cur_y_seq = y_seq[p3_indices]
        cur_s_seq = s_seq[p3_indices]
        cur_orig_idx = p3_indices
    else:
        cur_X_seq = X_seq
        cur_X_pool = X_pool
        cur_y_seq = y_seq
        cur_s_seq = s_seq
        cur_orig_idx = np.arange(N_total)
        
    N_samples = len(cur_y_seq)
    seed_records = []
    
    for seed in SEEDS:
        s_time = time.time()
        
        # 1. Setup Fold Splits for this Regime
        fold_splits = []  # list of (train_idx, test_idx)
        fold_assignments = np.zeros(N_samples, dtype=int)
        
        if regime == 'P1':
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for f, (tr, te) in enumerate(skf.split(cur_X_pool, cur_y_seq)):
                fold_splits.append((tr, te))
                fold_assignments[te] = f
                
        elif regime == 'P4':
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for f, (tr, te) in enumerate(sgkf.split(cur_X_pool, cur_y_seq, groups=cur_s_seq)):
                fold_splits.append((tr, te))
                fold_assignments[te] = f
                
        elif regime == 'P3':
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for f, (tr, te) in enumerate(skf.split(cur_X_pool, cur_y_seq)):
                fold_splits.append((tr, te))
                fold_assignments[te] = f
                
        elif regime == 'P2':
            # Within-subject temporal blocks with purged boundary
            subj_fold = np.zeros(N_samples, dtype=int)
            np.random.seed(seed)
            for s in np.unique(cur_s_seq):
                idx = np.where(cur_s_seq == s)[0]
                chunks = np.array_split(idx, 5)
                perm = np.random.permutation(5)
                for f, chunk in enumerate(chunks):
                    subj_fold[chunk] = perm[f]
            
            for f in range(5):
                te = np.where(subj_fold == f)[0]
                tr_raw = np.where(subj_fold != f)[0]
                # Purge temporal boundary within each subject
                purged_tr = []
                for s in np.unique(cur_s_seq):
                    s_te = te[cur_s_seq[te] == s]
                    s_tr = tr_raw[cur_s_seq[tr_raw] == s]
                    if len(s_te) == 0:
                        purged_tr.extend(s_tr)
                    else:
                        min_te = np.min(s_te)
                        max_te = np.max(s_te)
                        valid_s_tr = [i for i in s_tr if i < min_te - 4 or i > max_te + 4]
                        purged_tr.extend(valid_s_tr)
                fold_splits.append((np.array(purged_tr), te))
                fold_assignments[te] = f

        # Output probability arrays for all 7 models
        probs = {
            'decision_tree': np.zeros((N_samples, 3)),
            'svm_rbf': np.zeros((N_samples, 3)),
            'gradient_boosting': np.zeros((N_samples, 3)),
            'xgboost': np.zeros((N_samples, 3)),
            'random_forest': np.zeros((N_samples, 3)),
            'bilstm_44feat': np.zeros((N_samples, 3)),
            'dasi_msnet': np.zeros((N_samples, 3)),
        }
        
        preds = {m: np.zeros(N_samples, dtype=int) for m in probs}

        # Train and evaluate across the 5 folds
        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            X_tr_pool, y_tr = cur_X_pool[train_idx], cur_y_seq[train_idx]
            X_te_pool = cur_X_pool[test_idx]
            
            # --- 1. Decision Tree ---
            dt = DecisionTreeClassifier(max_depth=6, min_samples_split=4, random_state=seed + fold)
            dt.fit(X_tr_pool, y_tr)
            pr_dt = dt.predict_proba(X_te_pool)
            probs['decision_tree'][test_idx] = pr_dt
            preds['decision_tree'][test_idx] = pr_dt.argmax(axis=1)

            # --- 2. SVM (RBF Kernel) ---
            svm = make_pipeline(StandardScaler(), SVC(C=2.0, kernel='rbf', probability=True, random_state=seed + fold))
            svm.fit(X_tr_pool, y_tr)
            pr_svm = svm.predict_proba(X_te_pool)
            probs['svm_rbf'][test_idx] = pr_svm
            preds['svm_rbf'][test_idx] = pr_svm.argmax(axis=1)

            # --- 3. Gradient Boosting ---
            gb = GradientBoostingClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=seed + fold)
            gb.fit(X_tr_pool, y_tr)
            pr_gb = gb.predict_proba(X_te_pool)
            probs['gradient_boosting'][test_idx] = pr_gb
            preds['gradient_boosting'][test_idx] = pr_gb.argmax(axis=1)

            # --- 4. XGBoost (Das & Dev) ---
            xgb_clf = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, eval_metric='mlogloss', n_jobs=1, random_state=seed + fold)
            xgb_clf.fit(X_tr_pool, y_tr)
            pr_xgb = xgb_clf.predict_proba(X_te_pool)
            probs['xgboost'][test_idx] = pr_xgb
            preds['xgboost'][test_idx] = pr_xgb.argmax(axis=1)

            # --- 5. Random Forest ---
            rf = RandomForestClassifier(n_estimators=60, max_depth=6, n_jobs=1, random_state=seed + fold)
            rf.fit(X_tr_pool, y_tr)
            pr_rf = rf.predict_proba(X_te_pool)
            probs['random_forest'][test_idx] = pr_rf
            preds['random_forest'][test_idx] = pr_rf.argmax(axis=1)

            # --- 6. Standard Bi-LSTM (44 Feat) ---
            X_tr_seq_t = torch.tensor(cur_X_seq[train_idx], dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr, dtype=torch.long)
            X_te_seq_t = torch.tensor(cur_X_seq[test_idx], dtype=torch.float32)
            
            torch.manual_seed(seed + fold + 100)
            m_lstm = Standard_BiLSTM(in_features=44, hidden_dim=64, num_classes=3)
            opt_lstm = torch.optim.Adam(m_lstm.parameters(), lr=1e-3, weight_decay=1e-4)
            crit_ce = nn.CrossEntropyLoss()
            
            loader_lstm = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_tr_seq_t, y_tr_t), batch_size=32, shuffle=True
            )
            for ep in range(8):
                m_lstm.train()
                for bx, by in loader_lstm:
                    opt_lstm.zero_grad()
                    loss = crit_ce(m_lstm(bx), by)
                    loss.backward()
                    opt_lstm.step()
            m_lstm.eval()
            with torch.no_grad():
                out_lstm = m_lstm(X_te_seq_t)
                pr_lstm = F.softmax(out_lstm, dim=-1).numpy()
                probs['bilstm_44feat'][test_idx] = pr_lstm
                preds['bilstm_44feat'][test_idx] = pr_lstm.argmax(axis=1)

            # --- 7. DASI-MSNet (Ours) ---
            s_tr = cur_s_seq[train_idx]
            subj_map = {sid: i for i, sid in enumerate(np.unique(s_tr))}
            s_tr_mapped = torch.tensor([subj_map[sid] for sid in s_tr], dtype=torch.long)
            
            torch.manual_seed(seed + fold + 200)
            m_dasi = DASI_MSNet(in_features=44, num_subjects=len(subj_map), num_classes=3)
            m_dasi.set_lambda(0.5)
            opt_dasi = torch.optim.Adam(m_dasi.parameters(), lr=1e-3, weight_decay=1e-4)
            crit_focal = CostSensitiveOrdinalFocalLoss(gamma=2.0)
            crit_subj = nn.CrossEntropyLoss()
            
            loader_dasi = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_tr_seq_t, y_tr_t, s_tr_mapped), batch_size=32, shuffle=True
            )
            for ep in range(8):
                m_dasi.train()
                for bx, by, bs in loader_dasi:
                    opt_dasi.zero_grad()
                    y_out, s_out, _ = m_dasi(bx)
                    loss_task = crit_focal(y_out, by)
                    loss_s = crit_subj(s_out, bs)
                    (loss_task + 0.5 * loss_s).backward()
                    opt_dasi.step()
            m_dasi.eval()
            with torch.no_grad():
                y_out_te, _, _ = m_dasi(X_te_seq_t)
                pr_dasi = F.softmax(y_out_te, dim=-1).numpy()
                probs['dasi_msnet'][test_idx] = pr_dasi
                preds['dasi_msnet'][test_idx] = pr_dasi.argmax(axis=1)

        # Build DataFrame for this seed
        df_seed_dict = {
            'seed': [seed] * N_samples,
            'regime': [regime] * N_samples,
            'sequence_idx': cur_orig_idx,
            'subject_id': cur_s_seq,
            'fold': fold_assignments,
            'true_label': cur_y_seq,
            'true_label_name': [LABEL_NAMES[y] for y in cur_y_seq],
        }
        
        # Add columns for each model
        for m_name in ['decision_tree', 'svm_rbf', 'gradient_boosting', 'xgboost', 'bilstm_44feat', 'random_forest', 'dasi_msnet']:
            df_seed_dict[f'pred_{m_name}'] = preds[m_name]
            df_seed_dict[f'prob_{m_name}_c0'] = np.round(probs[m_name][:, 0], 4)
            df_seed_dict[f'prob_{m_name}_c1'] = np.round(probs[m_name][:, 1], 4)
            df_seed_dict[f'prob_{m_name}_c2'] = np.round(probs[m_name][:, 2], 4)
            
        df_s = pd.DataFrame(df_seed_dict)
        seed_records.append(df_s)
        
        acc_rf = accuracy_score(cur_y_seq, preds['random_forest']) * 100
        acc_lstm = accuracy_score(cur_y_seq, preds['bilstm_44feat']) * 100
        acc_dasi = accuracy_score(cur_y_seq, preds['dasi_msnet']) * 100
        print(f"  [{regime}] Seed {seed} done in {time.time()-s_time:.1f}s | Acc: RF={acc_rf:.2f}%, BiLSTM={acc_lstm:.2f}%, DASI={acc_dasi:.2f}%")

    # Combine 10 seeds for this regime
    df_regime = pd.concat(seed_records, ignore_index=True)
    regime_dfs[regime] = df_regime
    
    # Save regime CSV
    out_csv = os.path.join(REPO_ROOT, "data", f"oof_predictions_10seeds_{regime.lower()}.csv")
    df_regime.to_csv(out_csv, index=False)
    print(f"  [SAVED] {out_csv} ({len(df_regime)} rows)")
    print(f"  Regime {regime} finished in {time.time()-reg_start:.1f}s\n")

# Combine all 4 regimes into one master CSV
print(">>> COMBINING ALL 4 REGIMES INTO MASTER CSV <<<")
df_master = pd.concat(list(regime_dfs.values()), ignore_index=True)
master_csv = os.path.join(REPO_ROOT, "data", "oof_predictions_10seeds_7models.csv")
df_master.to_csv(master_csv, index=False)
print(f"Saved master multi-regime CSV: {master_csv} (Total rows: {len(df_master)})")

# Print Validation Summary Table across 10 seeds
print("\n" + "="*85)
print("10-SEED VALIDATION BENCHMARK SUMMARY (ACCURACY % MEAN ± SD)")
print("="*85)
models_order = [
    ('Decision Tree', 'decision_tree'),
    ('SVM (RBF Kernel)', 'svm_rbf'),
    ('Gradient Boosting', 'gradient_boosting'),
    ('XGBoost (Das & Dev)', 'xgboost'),
    ('Standard Bi-LSTM (44 Feat)', 'bilstm_44feat'),
    ('Random Forest', 'random_forest'),
    ('DASI-MSNet (Ours)', 'dasi_msnet'),
]

summary_rows = []
for label, m_key in models_order:
    row = {'Model': label}
    for reg in ['P1', 'P2', 'P3', 'P4']:
        df_r = regime_dfs[reg]
        seed_accs = []
        for s in SEEDS:
            sub_df = df_r[df_r['seed'] == s]
            acc = accuracy_score(sub_df['true_label'], sub_df[f'pred_{m_key}']) * 100
            seed_accs.append(acc)
        row[reg] = f"{np.mean(seed_accs):.2f}% ± {np.std(seed_accs):.2f}%"
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
print("="*85)
print("ALL RUNS AND CSVS GENERATED SUCCESSFULLY!")
