from data.dataset_loader import load_images
from utils.feature_utils import detect_and_compute
from utils.matching_utils import match_features
from utils.pnp_utils import estimate_camera_pose
from utils.triangulation_utils import triangulate_points
from utils.bundle_adjustment import bundle_adjustment
from visualization.plot_3d import plot_point_cloud
import numpy as np
import cv2
import os

# helper to filter matches
def filter_matches_geometric(kp1, kp2, matches):
    if len(matches) < 15:
        return matches
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # homography for flat wall
    H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers_h = np.sum(mask_h) if mask_h is not None else 0
    
    # fundamental matrix for corners
    F, mask_f = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 5.0, 0.99)
    inliers_f = np.sum(mask_f) if mask_f is not None else 0
    
    # If homography keeps > 70% of the matches that fundamental keeps
    # then the scene is likely flat so use homography
    # else the scene is a corner so use fundamental
    
    if inliers_h > (inliers_f * 0.7):
        # wall
        mask = mask_h
    else:
        # corner 
        mask = mask_f
        
    if mask is None: return []
    mask = mask.ravel()
    return [matches[i] for i in range(len(matches)) if mask[i]]

# helper to extract colours
def get_colors_from_keypoints(image, keypoints, matches, match_indices_mask=None):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    
    colors = []
    
    indices_to_process = range(len(matches))
    if match_indices_mask is not None:
        indices_to_process = np.where(match_indices_mask)[0]
    
    for i in indices_to_process:
        m = matches[i]
        pt = keypoints[m.queryIdx].pt
        
        # rounding to integer
        x, y = int(round(pt[0])), int(round(pt[1]))
        
        # boundary check
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        colors.append(img_rgb[y, x])
        
    return np.array(colors)

def save_ply(points, colors, filename="point_cloud.ply"):

    if colors is None:
        colors = np.ones_like(points) * 255
    
    colors = np.asarray(colors, dtype=np.uint8)
    points = np.asarray(points, dtype=np.float32)

    header = f"""ply
            format ascii 1.0
            element vertex {len(points)}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            """
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"Point cloud saved successfully to '{filename}'")

dataset_folder = r"PDC_Wall"
images = load_images(dataset_folder)
if not images: 
    exit()

K = np.array([[3222.22,    0.0, 1736.0],  
              [   0.0, 3222.22, 2320.0],  
              [   0.0,    0.0,    1.0]], dtype=np.float32) #found out by running exiftool.exe in terminal

init_idx1, init_idx2 = 0, 2 #hardcoded for good baseline
print(f"\nInitializing with Frame {init_idx1} and Frame {init_idx2}")

kp1, des1 = detect_and_compute(images[init_idx1])
kp2, des2 = detect_and_compute(images[init_idx2])
print(f"  Matching Frame {init_idx1} and Frame {init_idx2}...")
matches = match_features(des1, des2, ratio_test=True, ratio=0.80)
matches = filter_matches_geometric(kp1, kp2, matches)
print(f"  Verified Matches: {len(matches)}")

if len(matches) < 100:
    print("Error: Not enough matches between Frame 0 and 2.")
    exit()

# initial pose and triangulation

pts1_align = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2_align = np.float32([kp2[m.trainIdx].pt for m in matches])

E, mask = cv2.findEssentialMat(pts1_align, pts2_align, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R_init, t_init, _ = cv2.recoverPose(E, pts1_align, pts2_align, K)

P1_init = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P2_init = K @ np.hstack((R_init, t_init))

points_3d_init, mask_3d = triangulate_points(kp1, kp2, matches, K, P1_init, P2_init)

colors_init = get_colors_from_keypoints(images[init_idx1], kp1, matches, mask_3d)
colors_3d = colors_init

points_3d = points_3d_init[mask_3d]
matches = [matches[i] for i in range(len(matches)) if mask_3d[i]]
print(f"  Initial 3D Points Created: {len(points_3d)}")

# initial bundle adjustment
poses = [(np.eye(3), np.zeros((3,1))), (R_init, t_init)]
frame_to_pose_index = {init_idx1: 0, init_idx2: 1}
registered_frames = {init_idx1, init_idx2}

camera_indices = []
point_indices = []
points_2d = []

kp_to_3d_per_frame = {init_idx1: {}, init_idx2: {}}
all_kps = {init_idx1: kp1, init_idx2: kp2}
all_des = {init_idx1: des1, init_idx2: des2}

for i, m in enumerate(matches):
    camera_indices.append(0)
    point_indices.append(i)
    points_2d.append(kp1[m.queryIdx].pt)
    kp_to_3d_per_frame[init_idx1][m.queryIdx] = i
    
    camera_indices.append(1)
    point_indices.append(i)
    points_2d.append(kp2[m.trainIdx].pt)
    kp_to_3d_per_frame[init_idx2][m.trainIdx] = i

print("Running Initial Bundle Adjustment...")
poses_opt, points_3d_opt = bundle_adjustment(poses, points_3d, camera_indices, point_indices, points_2d, K)
poses = poses_opt
points_3d = points_3d_opt

debug_dir = "intermediate_results" 
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

# incremental sfm loop
for i in range(0, len(images)):
    if i in registered_frames: continue

    print(f"\nProcessing frame {i}...")
    kp_new, des_new = detect_and_compute(images[i])
    all_kps[i] = kp_new
    all_des[i] = des_new
    new_kp_to_3d = {}
    
    # matching against all currently registered frames to find best overlap
    best_ref_idx = -1
    best_matches = []
    
    # check top 3 closest registered frames
    sorted_reg = sorted(list(registered_frames), key=lambda x: abs(x-i))
    candidates = sorted_reg[:3]
    
    for ref_idx in candidates:
        curr_matches = match_features(des_new, all_des[ref_idx], ratio_test=True, ratio=0.80)
        curr_matches = filter_matches_geometric(kp_new, all_kps[ref_idx], curr_matches)
        
        if len(curr_matches) > len(best_matches):
            best_matches = curr_matches
            best_ref_idx = ref_idx

    if best_ref_idx == -1 or len(best_matches) < 20:
        print(f"  Skipping Frame {i}: No good matches found.")
        continue

    print(f"  Best Match: Frame {best_ref_idx} ({len(best_matches)} matches)")
    
    # PnP Preparation
    pnp_3d_points = []
    pnp_2d_points = []
    used_match_indices = set()
    
    for match_idx, m in enumerate(best_matches):
        if m.trainIdx in kp_to_3d_per_frame[best_ref_idx]:
            pt_3d_idx = kp_to_3d_per_frame[best_ref_idx][m.trainIdx]
            if pt_3d_idx < len(points_3d):
                pnp_3d_points.append(points_3d[pt_3d_idx])
                pnp_2d_points.append(kp_new[m.queryIdx].pt)
                new_kp_to_3d[m.queryIdx] = pt_3d_idx
                used_match_indices.add(match_idx)

    print(f"  PnP Candidates: {len(pnp_3d_points)}")

    if len(pnp_3d_points) < 8:
        print("  Not enough PnP candidates. Skipping.")
        kp_to_3d_per_frame[i] = {}
        continue

    R, t = estimate_camera_pose(pnp_3d_points, pnp_2d_points, K)
    
    if R is None:
        print(f"  PnP Failed for Frame {i}")
        kp_to_3d_per_frame[i] = {}
        continue
        
    print(f"  Frame {i} REGISTERED!")
    poses.append((R, t))
    frame_to_pose_index[i] = len(poses) - 1
    registered_frames.add(i)
    
    # adding BA observations (PnP points)
    pose_idx = len(poses) - 1
    for m in best_matches:
        if m.trainIdx in kp_to_3d_per_frame[best_ref_idx]:
            pt3d_idx = kp_to_3d_per_frame[best_ref_idx][m.trainIdx]
            if pt3d_idx < len(points_3d):
                camera_indices.append(pose_idx)
                point_indices.append(pt3d_idx)
                points_2d.append(kp_new[m.queryIdx].pt)

    # triangulate New Points
    triangulate_candidates = [m for idx, m in enumerate(best_matches) if idx not in used_match_indices]
    
    if len(triangulate_candidates) > 20:
        R_ref, t_ref = poses[frame_to_pose_index[best_ref_idx]]
        P_ref = K @ np.hstack((R_ref, t_ref))
        P_curr = K @ np.hstack((R, t))
        
        new_pts_3d, mask_valid = triangulate_points(kp_new, all_kps[best_ref_idx], triangulate_candidates, K, P_curr, P_ref)
        
        if np.sum(mask_valid) > 0:
            filtered_pts = new_pts_3d[mask_valid]
            start_idx = len(points_3d)
            points_3d = np.vstack([points_3d, filtered_pts])

            new_colors = get_colors_from_keypoints(images[i], kp_new, triangulate_candidates, mask_valid)
            colors_3d = np.vstack([colors_3d, new_colors])
            
            valid_indices = np.where(mask_valid)[0]
            for j, valid_idx in enumerate(valid_indices):
                m = triangulate_candidates[valid_idx]
                new_kp_to_3d[m.queryIdx] = start_idx + j
                
                # BA Obs
                camera_indices.append(pose_idx)
                point_indices.append(start_idx + j)
                points_2d.append(kp_new[m.queryIdx].pt)
                
                camera_indices.append(frame_to_pose_index[best_ref_idx])
                point_indices.append(start_idx + j)
                points_2d.append(all_kps[best_ref_idx][m.trainIdx].pt)
            
            print(f"  Triangulated {len(filtered_pts)} new points.")

    kp_to_3d_per_frame[i] = new_kp_to_3d

    # Local Bundle Adjustment
    if len(registered_frames) >= 4 and len(registered_frames) % 4 == 0:
        print("  Running Local Bundle Adjustment...")
        poses_opt, points_3d_opt = bundle_adjustment(poses, points_3d, camera_indices, point_indices, points_2d, K)
        poses = poses_opt
        points_3d = points_3d_opt

    if i % 4 == 0:
        print(f"  Saving point cloud at Frame {i}" )
        
        filename = os.path.join(debug_dir, f"reconstruction_frame_{i:03d}.ply")
        save_ply(points_3d, colors_3d, filename)

print("\nFinal Bundle Adjustment...")
poses_opt, points_3d_opt = bundle_adjustment(poses, points_3d, camera_indices, point_indices, points_2d, K)

save_ply(points_3d_opt, colors_3d, "reconstruction_result.ply")
plot_point_cloud(points_3d_opt, colors_3d)