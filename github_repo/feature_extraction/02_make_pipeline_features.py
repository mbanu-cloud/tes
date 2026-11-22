"""Step 3: convert data/wacv_3level_features.npz (from 01_extract_3level_features.py)
into data/features.npz, the input format read by pipeline/run_all.py."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = np.load(os.path.join(ROOT, 'data', 'wacv_3level_features.npz'), allow_pickle=True)
X, y, s = src['X_seq'], src['y_seq'], src['s_seq']
assert X.shape == (1044, 10, 44), X.shape
out = os.path.join(ROOT, 'data', 'features.npz')
np.savez(out, X=X.astype(np.float32), y=y, subject=s, sequence_idx=np.arange(len(y)))
print('saved', out, X.shape, np.bincount(y))
