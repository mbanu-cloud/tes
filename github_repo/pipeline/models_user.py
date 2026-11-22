"""Adaptor ke kode model ASLI penulis (experiments/dasi_msnet_architecture.py).

Loop training di sini menyalin persis generate_all_regimes_oof_predictions.py:
  - Bi-LSTM : torch.manual_seed(seed + fold + 100), Adam lr 1e-3 wd 1e-4, CE, batch 32, epoch tetap
  - DASI    : torch.manual_seed(seed + fold + 200), set_lambda(lambda), focal ordinal (tanpa class weight)
              + 0.5 * CE subjek, batch 32, epoch tetap
Dengan epoch = 8 dan seed yang sama, kolom bilstm_44feat dan dasi_msnet harus mereproduksi file OOF lama.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments')
sys.path.insert(0, ARCH_DIR)
from dasi_msnet_architecture import Standard_BiLSTM, Standard_TCN, DASI_MSNet, CostSensitiveOrdinalFocalLoss  # noqa: E402

SEED_OFFSET = {'bilstm': 100, 'tcn': 100, 'mstcn': 200}


def _latent_bilstm(m, x):
    out, _ = m.lstm(x)
    return out.mean(1)


def _latent_tcn(m, x):
    proj = m.input_proj(x).transpose(1, 2)
    out = F.gelu(m.conv1(proj))
    out = F.gelu(m.conv2(out)).transpose(1, 2)
    return m.norm(out).mean(1)


def _ev(m, X, y, multi):
    """Hanya untuk --epoch-test (diagnostik). Mengembalikan BA & akurasi uji per epoch tanpa mengubah RNG training."""
    if y is None:
        return {}
    from sklearn.metrics import balanced_accuracy_score
    was = m.training; m.eval()
    with torch.no_grad():
        out = m(X); out = out[0] if multi else out
    m.train(was)
    p = out.argmax(1).numpy()
    return {'test_ba': balanced_accuracy_score(y, p), 'test_acc': float((p == y).mean())}


def fit_predict(cfg_deep, spec, Xtr, ytr, str_, Xte, seed, fold, log_rows, tag, y_eval=None):
    """Latih satu model persis seperti skrip asli. Kembalikan (prob_test, model, info)."""
    torch.set_num_threads(cfg_deep.get('threads', 1))
    arch, epochs = spec['arch'], cfg_deep['max_epochs']
    d_in = Xtr.shape[2]
    X_tr_t = torch.tensor(Xtr, dtype=torch.float32)
    y_tr_t = torch.tensor(ytr, dtype=torch.long)
    X_te_t = torch.tensor(Xte, dtype=torch.float32)
    t0 = time.process_time()
    torch.manual_seed(seed + fold + SEED_OFFSET[arch])
    if arch in ('bilstm', 'tcn'):
        m = (Standard_BiLSTM if arch == 'bilstm' else Standard_TCN)(in_features=d_in, hidden_dim=64, num_classes=3)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
        for ep in range(1, epochs + 1):
            m.train(); tot = 0.0
            for bx, by in loader:
                opt.zero_grad(); loss = crit(m(bx), by); loss.backward(); opt.step(); tot += loss.item() * len(bx)
            log_rows.append({**tag, 'epoch': ep, 'train_loss': tot / len(X_tr_t), **_ev(m, X_te_t, y_eval, False)})
        m.eval()
        with torch.no_grad():
            prob = F.softmax(m(X_te_t), dim=-1).numpy()
    else:  # mstcn = DASI_MSNet dengan lambda tertentu (lambda 0 = tanpa GRL efektif)
        subj_map = {sid: i for i, sid in enumerate(np.unique(str_))}
        s_tr_t = torch.tensor([subj_map[s] for s in str_], dtype=torch.long)
        m = DASI_MSNet(in_features=d_in, num_subjects=len(subj_map), num_classes=3)
        m.set_lambda(spec['grl'])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        crit_task = CostSensitiveOrdinalFocalLoss(gamma=2.0) if spec['loss'] == 'focal' else nn.CrossEntropyLoss()
        crit_subj = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_t, y_tr_t, s_tr_t), batch_size=32, shuffle=True)
        for ep in range(1, epochs + 1):
            m.train(); tot = 0.0
            for bx, by, bs in loader:
                opt.zero_grad()
                y_out, s_out, _ = m(bx)
                loss = crit_task(y_out, by) + 0.5 * crit_subj(s_out, bs)
                loss.backward(); opt.step(); tot += loss.item() * len(bx)
            log_rows.append({**tag, 'epoch': ep, 'train_loss': tot / len(X_tr_t), **_ev(m, X_te_t, y_eval, True)})
        m.eval()
        with torch.no_grad():
            prob = F.softmax(m(X_te_t)[0], dim=-1).numpy()
    m._arch = arch
    return prob, m, {'epochs_run': epochs, 'cpu_s': time.process_time() - t0}


def embed(m, X):
    x = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        if m._arch == 'bilstm':
            return _latent_bilstm(m, x).numpy()
        if m._arch == 'tcn':
            return _latent_tcn(m, x).numpy()
        return m.extract_latent(x).numpy()
