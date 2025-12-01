import cv2
import numpy as np

def triangulate_points(kp1, kp2, matches, K, P1, P2):

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    
    # must be in front of the camera
    valid_mask = points_3d[:, 2] > 0
    
    # removing outliers
    if np.sum(valid_mask) > 10:
        current_valid_points = points_3d[valid_mask]
        
        center = np.median(current_valid_points, axis=0) 
        # distance of every point from the center
        distances = np.linalg.norm(points_3d - center, axis=1)
        valid_distances = distances[valid_mask]
        
        # keeping 95th percentile
        limit = np.percentile(valid_distances, 95)
        valid_mask &= (distances < limit)

    return points_3d, valid_mask