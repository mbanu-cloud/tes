"""
Module 01: Extract & Prepare 3-Level WACV Dataset
Directly processes WACV source images and feature alignments across all 3 classes:
- Class 0: Disengaged (from dataset/1 and processedData0.csv / merged_data0.csv)
- Class 1: Engaged / Partially Engaged (from dataset/2 and processedData1.csv / merged_data1.csv)
- Class 2: Highly Engaged / Fully Engaged (from dataset/3 and processedData2.csv / merged_data2.csv)

Outputs:
- Clean 16 Orthogonal Action Units
- Gaze & Pose Features
- Relative Facial Kinematic Velocities
- Frame-level & Micro-temporal Sequence Datasets
"""

import os
import glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WACV_DIR = os.path.join(BASE_DIR, "WACV data")
DATASET_DIR = os.path.join(WACV_DIR, "dataset")
OUT_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DATA_DIR, exist_ok=True)

def verify_raw_images():
    print("=== 1. VERIFYING RAW IMAGE SOURCE DATASET ===")
    folders = {'Class 0 (Disengaged)': os.path.join(DATASET_DIR, '1'),
               'Class 1 (Engaged)': os.path.join(DATASET_DIR, '2'),
               'Class 2 (Highly Engaged)': os.path.join(DATASET_DIR, '3')}
    
    total_imgs = 0
    for name, path in folders.items():
        if os.path.exists(path):
            imgs = glob.glob(os.path.join(path, "*.jpg"))
            print(f"  [OK] {name}: {len(imgs)} raw .jpg images found at: {path}")
            total_imgs += len(imgs)
        else:
            print(f"  [WARNING] Path not found: {path}")
    print(f"Total Raw Images Available: {total_imgs}\n")

def load_and_clean_3level_data():
    print("=== 2. LOADING & CLEANING 3-LEVEL FEATURE DATA ===")
    f0 = os.path.join(WACV_DIR, "merged_data0.csv")
    f1 = os.path.join(WACV_DIR, "merged_data1.csv")
    f2 = os.path.join(WACV_DIR, "merged_data2.csv")

    df0 = pd.read_csv(f0)
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)

    df0['Label'] = 0
    df1['Label'] = 1
    df2['Label'] = 2

    print(f"  Loaded Raw Frames -> Class 0: {len(df0)}, Class 1: {len(df1)}, Class 2: {len(df2)}")

    df = pd.concat([df0, df1, df2], ignore_index=True)
    df['Subject'] = df['ImageID'].apply(lambda x: int(x.split('_')[0].replace('img', '')))
    df['Frame'] = df['ImageID'].apply(lambda x: int(x.split('_')[1]))

    # IOD Normalization for AU15_r
    if 'x263' in df.columns and 'x33' in df.columns and 'AU15_r' in df.columns:
        dx = df['x263'] - df['x33']
        dy = df['y263'] - df['y33']
        dz = df['z263'] - df['z33'] if 'z263' in df.columns and 'z33' in df.columns else 0
        iod = np.sqrt(dx**2 + dy**2 + dz**2)
        iod_safe = np.where(iod > 1e-4, iod, 1.0)
        df['AU15_r'] = df['AU15_r'] / iod_safe
        print("  [OK] Applied Inter-Ocular Distance (IOD) normalization to AU15_r")

    # Select 16 Orthogonal AUs (dropping AU20_r and AU07_r)
    all_au = [c for c in df.columns if c.startswith('AU') and (c.endswith('_r') or c.endswith('_c'))]
    clean_aus = [c for c in all_au if c not in ['AU20_r', 'AU07_r']]
    print(f"  [OK] Selected 16 Orthogonal AUs: {clean_aus}")

    # Gaze & Pose Features
    gaze_pose_cols = [c for c in df.columns if 'gaze' in c or 'pose' in c]
    print(f"  [OK] Extracted {len(gaze_pose_cols)} Gaze & Pose features")

    feature_cols = clean_aus + gaze_pose_cols
    df = df.sort_values(['Subject', 'Frame']).reset_index(drop=True)

    # Compute Relative Kinematic Velocities: Delta v_t = p_t - p_{t-1}
    print("  [OK] Computing intra-subject relative kinematic velocities...")
    vel_cols = []
    for col in clean_aus:
        vel_col = f"vel_{col}"
        df[vel_col] = df.groupby('Subject')[col].diff().fillna(0.0)
        vel_cols.append(vel_col)

    final_feature_cols = feature_cols + vel_cols
    print(f"  Total Clean Feature Vector Dimension: {len(final_feature_cols)}")

    # Clean missing values
    df[final_feature_cols] = df[final_feature_cols].fillna(0.0)

    return df, final_feature_cols

def build_datasets(df, feature_cols, win_len=10, stride=3):
    print(f"\n=== 3. CONSTRUCTING MICRO-TEMPORAL & FRAME DATASETS (Window={win_len}, Stride={stride}) ===")
    
    # 1. Frame-level Dataset (3,307 frames)
    X_frame = df[feature_cols].values.astype(np.float32)
    y_frame = df['Label'].values.astype(np.int64)
    s_frame = df['Subject'].values.astype(np.int64)

    # 2. Sequence-level Dataset
    seq_list, seq_labels, seq_subjs = [], [], []
    for subj, grp in df.groupby('Subject'):
        grp = grp.sort_values('Frame')
        if len(grp) < win_len:
            continue
        for i in range(0, len(grp) - win_len + 1, stride):
            win = grp.iloc[i:i+win_len]
            seq_list.append(win[feature_cols].values.astype(np.float32))
            # Mode label for the window
            seq_labels.append(win['Label'].mode()[0])
            seq_subjs.append(subj)

    X_seq = np.array(seq_list, dtype=np.float32)
    y_seq = np.array(seq_labels, dtype=np.int64)
    s_seq = np.array(seq_subjs, dtype=np.int64)

    print("--- Frame-Level Dataset Summary ---")
    print(f"  Shape: {X_frame.shape}")
    print(f"  Class Distribution: {pd.Series(y_frame).value_counts().to_dict()}")

    print("--- Micro-Temporal Sequence Dataset Summary ---")
    print(f"  Shape: {X_seq.shape} (N_seq, T={win_len}, D={X_seq.shape[2]})")
    class_dist = pd.Series(y_seq).value_counts().to_dict()
    print(f"  Class Distribution: {class_dist}")
    print(f"  Active Unique Subjects: {len(np.unique(s_seq))}")

    # Save to NPZ
    out_file = os.path.join(OUT_DATA_DIR, "wacv_3level_features.npz")
    np.savez_compressed(
        out_file,
        X_frame=X_frame, y_frame=y_frame, s_frame=s_frame,
        X_seq=X_seq, y_seq=y_seq, s_seq=s_seq,
        feature_names=np.array(feature_cols)
    )
    print(f"\n[SUCCESS] Datasets saved to: {out_file}")

    # Save summary CSV
    summary_df = pd.DataFrame({
        'Level': [0, 1, 2],
        'Class_Name': ['Disengaged', 'Engaged', 'Highly Engaged'],
        'Frame_Count': [np.sum(y_frame == 0), np.sum(y_frame == 1), np.sum(y_frame == 2)],
        'Sequence_Count': [np.sum(y_seq == 0), np.sum(y_seq == 1), np.sum(y_seq == 2)],
        'Sequence_Percentage': [
            f"{np.sum(y_seq == 0)/len(y_seq)*100:.1f}%",
            f"{np.sum(y_seq == 1)/len(y_seq)*100:.1f}%",
            f"{np.sum(y_seq == 2)/len(y_seq)*100:.1f}%"
        ]
    })
    summary_df.to_csv(os.path.join(OUT_DATA_DIR, "dataset_3level_summary.csv"), index=False)
    print(summary_df)

if __name__ == "__main__":
    verify_raw_images()
    df, feature_cols = load_and_clean_3level_data()
    build_datasets(df, feature_cols, win_len=10, stride=3)
