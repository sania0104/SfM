import numpy as np
import cv2

def estimate_camera_pose(points_3d, points_2d, K):

    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)

    if len(points_3d) < 4 or len(points_2d) < 4:
        print(f"Not enough points for PnP: {len(points_3d)} points. Skipping frame.")
        return None, None

    # with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, K, None,
        reprojectionError=20.0,
        confidence=0.99,
        iterationsCount=1000,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        # Fallback to Iterative
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            reprojectionError=20.0, 
            confidence=0.99,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
    if rvec is None:
        print("PnP failed for this frame. Reason: rvec is none")
        return None, None
    if tvec is None:
        print("PnP failed for this frame. Reason: tvec is none")
        return None, None
    
    if inliers is not None and len(inliers) < 5:
        print(f"PnP succeeded but too few inliers ({len(inliers)}). Rejecting.")
        return None, None

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec
    return R, t