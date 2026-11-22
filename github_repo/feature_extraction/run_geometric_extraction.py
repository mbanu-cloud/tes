import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import sys

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helper for AU calculation
try:
    from calculate_geometric_au import euclidean_dist, calculate_ear
except ImportError:
    # If import fails, define them here
    def euclidean_dist(df, idx1, idx2):
        return np.sqrt(
            (df[f'x{idx1}'] - df[f'x{idx2}'])**2 + 
            (df[f'y{idx1}'] - df[f'y{idx2}'])**2
        )

    def calculate_ear(df, indices):
        p1, p2, p3, p4, p5, p6 = indices
        dist_vertical1 = euclidean_dist(df, p2, p6)
        dist_vertical2 = euclidean_dist(df, p3, p5)
        dist_horizontal = euclidean_dist(df, p1, p4)
        ear = (dist_vertical1 + dist_vertical2) / (2.0 * dist_horizontal + 1e-6)
        return ear

def calculate_geometric_aus(df):
    """
    Calculates Geometric Action Units (GAUs) from MediaPipe landmarks.
    Expects df with columns x0, y0, ..., x467, y467.
    Returns df with added GAU columns.
    """
    # 1. Inter-Ocular Distance (IOD) for Normalization
    # Using Iris Centers: Left 468, Right 473 (MediaPipe Iris) - BUT 468 points model doesn't have iris?
    # Standard 468 model: 
    # Left Eye Center approx: 468 is usually iris center in 478 model.
    # In 468 model, use eye corners: Left(33, 133), Right(362, 263).
    # Center Left Eye: (x33+x133)/2. Center Right Eye: (x362+x263)/2.
    # Distance between centers.
    
    # Let's use Eye Corners for IOD: Outer Left (33) to Outer Right (263)
    # Or Inner Left (133) to Inner Right (362).
    # Common practice: Outer corners.
    iod = euclidean_dist(df, 33, 263)
    iod = np.where(iod == 0, 1.0, iod)
    
    # Initialize GAU columns
    # We use names compatible with ML_classification.py if possible, 
    # but ML_classification expects AU01_r, etc.
    # We will map GAUs to those names for compatibility.
    
    # --- AU1: Inner Brow Raiser ---
    d_left = euclidean_dist(df, 107, 66) # Brow(107) to Eye(66)? No, 107 is brow, 66 is...
    # Using the logic from calculate_geometric_au.py if available, or standard:
    # AU1: Inner Brow (107, 336) to Inner Eye Corner (133, 362)? 
    # Let's use the provided logic in calculate_geometric_au.py snippet:
    # AU1: d(107, 66)?? 66 is near inner eye? 
    # Let's just use standard mappings.
    
    # AU1 (Inner Brow Raiser): Inner Brow (107/336) height relative to eye.
    # d(107, 133) + d(336, 362)
    df['AU01_r'] = (euclidean_dist(df, 107, 133) + euclidean_dist(df, 336, 362)) / (2.0 * iod)

    # AU2 (Outer Brow Raiser): Outer Brow (46/276) height relative to eye.
    # d(46, 33) + d(276, 263)
    df['AU02_r'] = (euclidean_dist(df, 46, 33) + euclidean_dist(df, 276, 263)) / (2.0 * iod)

    # AU4 (Brow Lowerer): Distance between brows (107, 336) and brow to eye.
    # d(107, 336)
    df['AU04_r'] = euclidean_dist(df, 107, 336) / iod

    # AU5 (Upper Lid Raiser): Upper Eyelid (159/386) to Eye Center/LowerLid(145/374)
    # d(159, 145) + d(386, 374)
    df['AU05_r'] = (euclidean_dist(df, 159, 145) + euclidean_dist(df, 386, 374)) / (2.0 * iod)

    # AU6 (Cheek Raiser): Cheek (123/352) to Eye Outer Corner (33/263)? Or 133/362?
    # d(123, 133) + d(352, 362) (Cheek to Inner corner? Maybe Outer is better: 33/263)
    # Using previous logic:
    df['AU06_r'] = (euclidean_dist(df, 123, 133) + euclidean_dist(df, 352, 362)) / (2.0 * iod)

    # AU7 (Lid Tightener): Inverse of EAR (Squinting)
    # We can use EAR.
    ear_left = calculate_ear(df, [33, 160, 158, 133, 153, 144])
    ear_right = calculate_ear(df, [362, 385, 387, 263, 373, 380])
    df['AU07_r'] = 1.0 - ((ear_left + ear_right) / 2.0)

    # AU9 (Nose Wrinkler): Nose Tip (4) to Bridge (197)
    df['AU09_r'] = euclidean_dist(df, 4, 197) / iod

    # AU10 (Upper Lip Raiser): Upper Lip (0) to Nose Base (2)
    df['AU10_r'] = euclidean_dist(df, 0, 2) / iod

    # AU12 (Lip Corner Puller): Lip Corner Distance (61, 291)
    df['AU12_r'] = euclidean_dist(df, 61, 291) / iod

    # AU14 (Dimpler): Corner (61/291) to Cheek (50/280)
    df['AU14_r'] = (euclidean_dist(df, 61, 50) + euclidean_dist(df, 291, 280)) / (2.0 * iod)

    # AU15 (Lip Corner Depressor): Corner height relative to center
    # d(Corner, Chin) - d(Center, Chin)
    d_center_chin = euclidean_dist(df, 17, 152)
    d_left_chin = euclidean_dist(df, 61, 152)
    d_right_chin = euclidean_dist(df, 291, 152)
    df['AU15_r'] = ((d_left_chin + d_right_chin)/2.0) - d_center_chin

    # AU17 (Chin Raiser): Chin (175) to Lower Lip (17)
    df['AU17_r'] = euclidean_dist(df, 175, 17) / iod

    # AU20 (Lip Stretcher): Horizontal stretch (similar to AU12 but wider?)
    # Just use AU12 metric for now or skip.
    df['AU20_r'] = df['AU12_r'] 

    # AU23 (Lip Tightener): Lip rolling in. d(0, 13) + d(17, 14)
    df['AU23_r'] = (euclidean_dist(df, 0, 13) + euclidean_dist(df, 17, 14)) / iod

    # AU25 (Lips Part): Upper Lip (0) to Lower Lip (17)
    df['AU25_r'] = euclidean_dist(df, 0, 17) / iod

    # AU26 (Jaw Drop): Upper Lip (0) to Chin (152)
    df['AU26_r'] = euclidean_dist(df, 0, 152) / iod

    # AU45 (Blink): Inverse EAR (similar to AU7 but specifically closing)
    # We use EAR directly? Or 1 - EAR.
    # Usually AU45 is "Blink", so 1 - EAR is a good proxy for "closedness".
    df['AU45_r'] = 1.0 - ((ear_left + ear_right) / 2.0)
    
    # Add dummy 'AU45_c' (Presence) for compatibility
    df['AU45_c'] = np.where(df['AU45_r'] > 0.8, 1, 0) # Threshold

    return df

def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates Head Pose (Tx, Ty, Tz, Rx, Ry, Rz) using SolvePnP.
    Returns: dictionary with pose parameters.
    """
    # Define 3D Model Points (Generic Face)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    # 2D Image Points from Landmarks
    # Nose(1), Chin(152), LeftEye(33), RightEye(263), LeftMouth(61), RightMouth(291)
    # Note: landmarks are normalized, need to scale to img_w, img_h
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
        (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
        (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye left corner
        (landmarks[263].x * img_w, landmarks[263].y * img_h), # Right eye right corner
        (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left Mouth corner
        (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth corner
    ], dtype=np.float64)

    # Camera Internals (Approximate)
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return {
            'pose_Tx': 0, 'pose_Ty': 0, 'pose_Tz': 0,
            'pose_Rx': 0, 'pose_Ry': 0, 'pose_Rz': 0
        }

    # Rotation Vector is essentially the axis-angle representation (radians)
    # OpenFace pose_Rx, Ry, Rz are typically Euler angles in radians.
    # But usually, just using the rotation vector components is a good enough feature representation.
    # However, to be precise, let's keep them as the rotation vector components directly 
    # as they represent the rotation around X, Y, Z axes.
    
    return {
        'pose_Tx': translation_vector[0][0],
        'pose_Ty': translation_vector[1][0],
        'pose_Tz': translation_vector[2][0],
        'pose_Rx': rotation_vector[0][0],
        'pose_Ry': rotation_vector[1][0],
        'pose_Rz': rotation_vector[2][0]
    }

def get_gaze_vector(landmarks, img_w, img_h):
    """
    Estimates Gaze Vector (gaze_0_x, y, z for Left Eye, gaze_1_x, y, z for Right Eye).
    Uses Iris Center relative to Eye Center.
    Requires refine_landmarks=True.
    """
    # Helper to get numpy array from landmark
    def to_np(lm):
        return np.array([lm.x * img_w, lm.y * img_h, lm.z * img_w]) # z is relative to width roughly
    
    # Left Eye: 33 (Outer), 133 (Inner). Center approx.
    # Left Iris Center: 468
    p_left_outer = to_np(landmarks[33])
    p_left_inner = to_np(landmarks[133])
    p_left_center = (p_left_outer + p_left_inner) / 2.0
    p_left_iris = to_np(landmarks[468])
    
    v_left = p_left_iris - p_left_center
    norm_left = np.linalg.norm(v_left)
    if norm_left > 0:
        v_left /= norm_left
    else:
        v_left = np.array([0.0, 0.0, 1.0]) # Default forward

    # Right Eye: 362 (Inner), 263 (Outer). Center approx.
    # Right Iris Center: 473
    p_right_inner = to_np(landmarks[362])
    p_right_outer = to_np(landmarks[263])
    p_right_center = (p_right_inner + p_right_outer) / 2.0
    p_right_iris = to_np(landmarks[473])
    
    v_right = p_right_iris - p_right_center
    norm_right = np.linalg.norm(v_right)
    if norm_right > 0:
        v_right /= norm_right
    else:
        v_right = np.array([0.0, 0.0, 1.0])

    return {
        'gaze_0_x': v_left[0], 'gaze_0_y': v_left[1], 'gaze_0_z': v_left[2],
        'gaze_1_x': v_right[0], 'gaze_1_y': v_right[1], 'gaze_1_z': v_right[2]
    }

def process_folder(folder_path, label, output_path):
    print(f"Processing folder: {folder_path} for Label {label}")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7
    )
    
    data = []
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images.")
    
    for file in files:
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract 468 landmarks
                row = {
                    'ImageID': file.split('.')[0],
                    'Label_y': label, # Mapping to Label_y for compatibility
                    'confidence': 1.0 # Mock confidence for filter
                }
                
                ih, iw, _ = image.shape
                
                # Add x_0..x_467, y_0..y_467
                for i, lm in enumerate(face_landmarks.landmark):
                    if i < 468: # Limit to 468
                        row[f'x_{i}'] = int(lm.x * iw)
                        row[f'y_{i}'] = int(lm.y * ih)

                # Calculate S2: Head Pose & Gaze
                pose = get_head_pose(face_landmarks.landmark, iw, ih)
                row.update(pose)
                
                # Check if we have iris landmarks (468-477)
                if len(face_landmarks.landmark) >= 478:
                    gaze = get_gaze_vector(face_landmarks.landmark, iw, ih)
                    row.update(gaze)
                else:
                    # Mock Gaze if not available
                    for k in ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']:
                        row[k] = 0.0
                
                data.append(row)
    
    if not data:
        print("No faces detected in this folder.")
        return

    df = pd.DataFrame(data)
    
    # Calculate Geometric AUs
    print("Calculating Geometric AUs...")
    df = calculate_geometric_aus(df)
    
    # Rename columns x_0 -> x0 for ML compatibility
    rename_dict = {}
    for i in range(468):
        rename_dict[f'x_{i}'] = f'x{i}'
        rename_dict[f'y_{i}'] = f'y{i}'
    df = df.rename(columns=rename_dict)

    # Reorder columns to ensure x0..x467 then y0..y467
    x_cols = [f'x{i}' for i in range(468)]
    y_cols = [f'y{i}' for i in range(468)]
    
    # Other columns we want to keep
    # We need to make sure we keep ImageID, Label_y, confidence, AUs, S2 columns
    # Let's list the AU columns we created
    au_cols = [c for c in df.columns if c.startswith('AU') or c.startswith('gaze') or c.startswith('pose') or c in ['ImageID', 'Label_y', 'confidence']]
    
    # gaze_0_x..z, gaze_1_x..z, pose_Tx..Rz
    s2_cols = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 
               'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
               'pose_Tx', 'pose_Ty', 'pose_Tz',
               'pose_Rx', 'pose_Ry', 'pose_Rz']
    
    # Check if S2 columns exist (they should be added in the loop)
    # If not (e.g. error in calculation), fill with 0
    for col in s2_cols:
        if col not in df.columns:
            df[col] = 0.0
        
    # Reorder columns: ImageID, Label_y, confidence, x0..x467, y0..y467, S2, AUs
    # Ensure x0..x467 then y0..y467 are contiguous for ML_classification slicing
    # Ensure S2 is included in the slice x0:AU45_c (so S2 must be before AUs)
    x_cols = [f'x{i}' for i in range(468)]
    y_cols = [f'y{i}' for i in range(468)]
    
    au_cols = [c for c in df.columns if c.startswith('AU')]
    # Sort AU cols
    au_cols.sort()
    # Ensure AU45_c is strictly the LAST column among AUs, because train_model_ML.py slices up to AU45_c
    # and we need all other AUs (including AU45_r) to be included in that slice.
    if 'AU45_c' in au_cols:
        au_cols.remove('AU45_c')
        au_cols.append('AU45_c')
    
    other_cols = [c for c in df.columns if c not in x_cols and c not in y_cols and c not in s2_cols and c not in au_cols and c not in ['ImageID', 'Label_y', 'confidence']]
    
    final_cols = ['ImageID', 'Label_y', 'confidence'] + x_cols + y_cols + s2_cols + au_cols + other_cols
    
    # Filter only columns that exist
    final_cols = [c for c in final_cols if c in df.columns]
    
    df = df[final_cols]
        
    # Save to CSV
    output_path = output_path.replace("processedData", "merged_data")
    print(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")

def main():
    # Set this to your dataset folder
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(REPO_ROOT, "WACV data", "dataset") 
    
    # Process each label folder
    # Label 0: 0 and 1 combined (from original DAiSEE/WACV 0,1,2,3 mapped to 0,1,2)
    # But here we are processing folders 1, 2, 3 directly from WACV dataset structure.
    # WACV structure usually: 1 (Low?), 2 (High?), 3 (Very High?)
    # Let's assume folder names 1, 2, 3 map to labels 0, 1, 2.
    
    # Folder "1" -> Label 0
    process_folder(os.path.join(base_path, "1"), 0, os.path.join(REPO_ROOT, "WACV data", "merged_data0.csv"))
    
    # Folder "2" -> Label 1
    process_folder(os.path.join(base_path, "2"), 1, os.path.join(REPO_ROOT, "WACV data", "merged_data1.csv"))
    
    # Folder "3" -> Label 2
    process_folder(os.path.join(base_path, "3"), 2, os.path.join(REPO_ROOT, "WACV data", "merged_data2.csv"))

if __name__ == "__main__":
    main()
