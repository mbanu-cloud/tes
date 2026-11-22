# Identity or Overlap? A Four-Regime Leakage Audit of Student Engagement Recognition on WACV

Code, frozen folds and out-of-fold predictions for the manuscript
*"Identity or Overlap? A Four-Regime Leakage Audit of Facial-Geometry Student Engagement Recognition on the WACV Benchmark"*
(N. Wibisono et al., submitted to the Nigerian Journal of Technology).

Every number in the manuscript can be regenerated from the files in `results/` without retraining.
Retraining from raw images is also possible (Section "Full reproduction").

## Repository layout

| Folder | Content |
|---|---|
| `feature_extraction/` | MediaPipe FaceMesh landmarks → 16 geometric AUs, 12 gaze/pose values, 16 AU velocities; windows of T = 10 frames, stride 3 |
| `experiments/` | DASI-MSNet, Bi-LSTM and TCN definitions; script that trains the 7 models in the 4 regimes × 10 seeds |
| `pipeline/` | Frozen folds (`folds.csv`), configs, single runner for deep models / ablation / identity probe, verification and statistics |
| `analysis/` | Within-subject AUC and the scripts that draw Figures 1–3 and 5–7 |
| `data/` | Out-of-fold predictions of the 7 models (4 regimes × 10 seeds). `features.npz` is **not** distributed (see Data) |
| `results/main_10seeds/` | Main run: deep models, 8-configuration ablation, identity probes, CPU time, training logs, statistics |
| `results/sensitivity_30epochs/` | 30-epoch sensitivity analysis (P1, P4) |
| `results/epoch_test_100/` | 100-epoch convergence check (P1, seed 42) |

## Where each result comes from

| Manuscript item | File(s) |
|---|---|
| Table 2, Figure 2 (regimes, leakage decomposition) | `results/main_10seeds/results.json` (`seed`, `decomp`) |
| Table 3, Figure 3 (P4 performance, permutation test) | `results.json` (`ens_p4`), `nulls.npz` |
| Table 4, Figures 5–6 | `results.json`, `ens_probs.npz`, `probe.csv`, `timing.csv` |
| Section 4.4 (within-subject AUC) | `analysis/within_subject_auc.py` |
| Table 5, Figure 7 (ablation, identity probe) | `results/main_10seeds/all.csv` (columns `abl_c1`–`abl_c8`), `probe.csv` |
| Section 4.8 (epoch sensitivity) | `results/sensitivity_30epochs/`, `results/epoch_test_100/` |

## Quick check (no training, ~2 min)

```bash
pip install -r requirements.txt
cd pipeline
python verify_results.py ../results/main_10seeds --classical-from ../data/oof_predictions_10seeds_7models.csv
python stats_all.py ../results/main_10seeds          # recomputes results.json
cd .. && python analysis/within_subject_auc.py && python analysis/make_figures_2_3_5_6.py && python analysis/make_figure_7.py
```

`verify_results.py` checks that the deep-model predictions are identical to the original 7-model run,
that ablation configurations 2 and 7 reproduce the Bi-LSTM and DASI-MSNet predictions exactly,
that P4 is subject-disjoint, and that all folds match `pipeline/folds.csv`.

## Full reproduction from raw images

1. Download the WACV 2016 engagement dataset (Kamath et al.) from <https://github.com/e-drishti/wacv2016>
   and place it as `WACV data/dataset/` in the repository root.
2. Extract features and build sequences:
   ```bash
   python feature_extraction/run_geometric_extraction.py     # -> WACV data/merged_data{0,1,2}.csv
   python feature_extraction/01_extract_3level_features.py   # -> data/wacv_3level_features.npz (1,044 sequences)
   python feature_extraction/02_make_pipeline_features.py    # -> data/features.npz
   ```
3. Train the 7 models in the 4 regimes × 10 seeds (classical models and main deep models):
   ```bash
   python experiments/generate_all_regimes_oof_predictions.py   # -> data/oof_predictions_10seeds_7models.csv
   ```
4. Deep models, ablation, identity probes and CPU time with the frozen folds:
   ```bash
   cd pipeline
   python run_all.py --skip-classical                                   # main run (8 epochs)
   python run_all.py --config config_sens30.yaml --skip-classical      # 30-epoch sensitivity
   python run_all.py --epoch-test                                       # 100-epoch convergence check
   python verify_results.py ../results/<new_folder> --classical-from ../data/oof_predictions_10seeds_7models.csv
   python stats_all.py ../results/<new_folder>
   ```

Exact reproduction of the deep-model predictions was obtained with Python 3.9, PyTorch 2.2.2 and
scikit-learn 1.6.1 on a single CPU thread (macOS 14, x86_64). Other versions or platforms may change
the deep-model predictions slightly; the classical models and all statistics are platform-independent.

## Data

The WACV images belong to their original authors and are not redistributed. The derived feature file
`features.npz` is not included either; it is produced by steps 1–2 above. The released out-of-fold
predictions and statistics contain no images or facial features.

## Citation

If you use this code, please cite the manuscript (citation will be added after publication) and the dataset:
A. Kamath, A. Biswas, V. Balasubramanian, "A Crowdsourced Approach to Student Engagement Recognition in
e-Learning Environments," IEEE WACV, 2016. https://doi.org/10.1109/WACV.2016.7477618

## License

Code: MIT (see `LICENSE`). Results files: CC BY 4.0.
