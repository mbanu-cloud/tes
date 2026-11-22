
import pandas as pd
import numpy as np
import os

def euclidean_dist(df, idx1, idx2):
    return np.sqrt(
        (df[f'x_{idx1}'] - df[f'x_{idx2}'])**2 + 
        (df[f'y_{idx1}'] - df[f'y_{idx2}'])**2
    )

def calculate_ear(df, indices):
    # indices: [p1, p2, p3, p4, p5, p6]
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    p1, p2, p3, p4, p5, p6 = indices
    
    dist_vertical1 = euclidean_dist(df, p2, p6)
    dist_vertical2 = euclidean_dist(df, p3, p5)
    dist_horizontal = euclidean_dist(df, p1, p4)
    
    # Avoid division by zero
    ear = (dist_vertical1 + dist_vertical2) / (2.0 * dist_horizontal + 1e-6)
    return ear

def calculate_geometric_aus(file_path):
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Inter-Ocular Distance (IOD) for Normalization
    # Using Iris Centers: Left 468, Right 473
    iod = euclidean_dist(df, 468, 473)
    
    # Avoid division by zero
    iod = np.where(iod == 0, 1.0, iod)
    
    # Initialize Result DataFrame
    geo_au = pd.DataFrame()
    geo_au['Frame'] = df.index
    
    # --- AU1: Inner Brow Raiser ---
    d_left = euclidean_dist(df, 107, 133)
    d_right = euclidean_dist(df, 336, 362)
    geo_au['GAU1_InnerBrow'] = (d_left + d_right) / (2.0 * iod)
    
    # --- AU2: Outer Brow Raiser ---
    d_left = euclidean_dist(df, 46, 33)
    d_right = euclidean_dist(df, 276, 263)
    geo_au['GAU2_OuterBrow'] = (d_left + d_right) / (2.0 * iod)
    
    # --- AU4: Brow Lowerer ---
    geo_au['GAU4_BrowDist'] = euclidean_dist(df, 107, 336) / iod
    
    # --- AU5: Upper Lid Raiser (Upper EAR) ---
    # Upper Lid to Center Eye: Left(159-158), Right(386-387) ? Approx with EAR_Upper
    # Better: Dist(UpperLid, EyeCenter)
    geo_au['GAU5_UpperLid'] = (euclidean_dist(df, 159, 145) + euclidean_dist(df, 386, 374)) / (2.0 * iod)

    # --- AU6: Cheek Raiser ---
    # Cheek (123/352) to Eye Corner (133/362)
    d_left = euclidean_dist(df, 123, 133)
    d_right = euclidean_dist(df, 352, 362)
    geo_au['GAU6_CheekRaise'] = (d_left + d_right) / (2.0 * iod)

    # --- AU7: Lid Tightener (Squint) ---
    # Similar to EAR but specifically measuring narrowness without full blink
    # We use inverse EAR as proxy for tightness
    ear_left = calculate_ear(df, [33, 160, 158, 133, 153, 144])
    ear_right = calculate_ear(df, [362, 385, 387, 263, 373, 380])
    geo_au['GAU7_LidTight'] = 1.0 - ((ear_left + ear_right) / 2.0)

    # --- AU9: Nose Wrinkler ---
    # Dist Nose Tip (4) to Nose Bridge (197)
    geo_au['GAU9_NoseWrinkle'] = euclidean_dist(df, 4, 197) / iod

    # --- AU10: Upper Lip Raiser ---
    # Upper Lip Top (0) to Nose Base (2)
    geo_au['GAU10_LipRaise'] = euclidean_dist(df, 0, 2) / iod

    # --- AU12: Lip Corner Puller ---
    geo_au['GAU12_LipWidth'] = euclidean_dist(df, 61, 291) / iod

    # --- AU14: Dimpler ---
    # Lip Corner (61/291) to Cheek Outer (50/280)
    # When dimpling, this distance usually decreases or changes
    d_left = euclidean_dist(df, 61, 50)
    d_right = euclidean_dist(df, 291, 280)
    geo_au['GAU14_Dimpler'] = (d_left + d_right) / (2.0 * iod)

    # --- AU15: Lip Corner Depressor ---
    # Lip Corner (61/291) vertical position relative to Lip Center (0/17)
    # We measure if corners are lower than center.
    # Simple metric: Dist(Corner, Chin) - Dist(Center, Chin)
    # If Corner is closer to chin -> Frown
    d_center_chin = euclidean_dist(df, 17, 152)
    d_left_chin = euclidean_dist(df, 61, 152)
    d_right_chin = euclidean_dist(df, 291, 152)
    geo_au['GAU15_Frown'] = ((d_left_chin + d_right_chin)/2.0) - d_center_chin

    # --- AU17: Chin Raiser ---
    # Chin (175) to Lower Lip (17)
    geo_au['GAU17_ChinRaise'] = euclidean_dist(df, 175, 17) / iod

    # --- AU20: Lip Stretcher ---
    # Similar to AU12 but more horizontal stretch
    # We use same metric as AU12 here as proxy
    geo_au['GAU20_LipStretch'] = geo_au['GAU12_LipWidth']

    # --- AU23: Lip Tightener ---
    # Lip Thickness (Upper 0-13 vs Lower 14-17)
    # Or simply vertical distance of lips when closed.
    # Proxy: Inverse of Mouth Openness when mouth is closed?
    # Hard to measure compression geometrically without depth.
    # We use width/height ratio of mouth.
    mouth_h = euclidean_dist(df, 0, 17) + 1e-6
    mouth_w = euclidean_dist(df, 61, 291)
    geo_au['GAU23_LipTight'] = mouth_w / mouth_h

    # --- AU25: Lips Part ---
    geo_au['GAU25_MouthOpen'] = euclidean_dist(df, 0, 17) / iod
    
    # --- AU26: Jaw Drop ---
    # Same metric as AU25, but usually thresholded higher.
    geo_au['GAU26_JawDrop'] = geo_au['GAU25_MouthOpen']

    # --- AU45: Blink ---
    geo_au['GAU45_EAR'] = (ear_left + ear_right) / 2.0
    
    return geo_au, df

if __name__ == "__main__":
    # Path to the sample file found
    FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "WACV data", "dataset", "processed1", "img22_79.csv")
    
    if os.path.exists(FILE_PATH):
        geo_au_df, original_df = calculate_geometric_aus(FILE_PATH)
        
        print("\n=== GEOMETRIC ACTION UNITS (GAUs) CALCULATED FROM MEDIAPIPE ===")
        print(geo_au_df.head(10).to_string(index=False))
        
        print("\n=== COMPARISON WITH ORIGINAL OPENFACE AUs (Subset) ===")
        comparison = pd.DataFrame({
            'GAU1 (Ours)': geo_au_df['GAU1_InnerBrow'],
            'AU01_r (OpenFace)': original_df['AU01_r'],
            'GAU12 (Ours)': geo_au_df['GAU12_LipWidth'],
            'AU12_r (OpenFace)': original_df['AU12_r'],
            'GAU25 (Ours)': geo_au_df['GAU25_MouthOpen'],
            'AU25_r (OpenFace)': original_df['AU25_r']
        })
        print(comparison.head(10).to_string(index=False))
        
        print("\n[INFO] These GAUs are calculated purely from 'x_' and 'y_' coordinates using the formulas in Table 1.")
    else:
        print(f"File not found: {FILE_PATH}")
