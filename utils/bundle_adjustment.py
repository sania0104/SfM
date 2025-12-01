import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def reprojection_residual(params, n_cams, n_points, camera_indices, point_indices, points_2d, K):

    camera_params = params[:n_cams * 6].reshape((n_cams, 6))
    points_3d = params[n_cams * 6:].reshape((n_points, 3))
    
    points_3d_obs = points_3d[point_indices]
    cameras_obs = camera_params[camera_indices]

    residuals = np.zeros(len(points_2d) * 2)
    
    for i in range(len(points_2d)):
        rvec = cameras_obs[i, :3]
        tvec = cameras_obs[i, 3:6]
        pt3d = points_3d_obs[i]
    
        proj_pt, _ = cv2.projectPoints(pt3d.reshape(1, 1, 3), rvec, tvec, K, None)
        proj_pt = proj_pt.ravel()
        
        observed_pt = points_2d[i]
        residuals[2*i] = proj_pt[0] - observed_pt[0]
        residuals[2*i+1] = proj_pt[1] - observed_pt[1]
        
    return residuals

def bundle_adjustment_sparsity(n_cams, n_points, camera_indices, point_indices):

    m = len(camera_indices) * 2  # number of residuals (x and y for each observation)
    n = n_cams * 6 + n_points * 3 # number of parameters
    
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(len(camera_indices))

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
        
    for s in range(3):
        A[2 * i, n_cams * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cams * 6 + point_indices * 3 + s] = 1
        
    return A

def bundle_adjustment(cameras, points_3d, camera_indices, point_indices, points_2d, K):

    n_cams = len(cameras)
    n_points = points_3d.shape[0]
    n_observations = len(points_2d)
    
    camera_params = []
    for R, t in cameras:
        rvec, _ = cv2.Rodrigues(R)
        camera_params.append(np.hstack([rvec.ravel(), t.ravel()]))
    camera_params = np.array(camera_params).reshape(-1)
    
    # flatten 3D points
    points_3d_flat = points_3d.ravel()
    
    x0 = np.hstack([camera_params, points_3d_flat])
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)

    print(f"  Constructing Sparsity Matrix for {n_observations} observations...")
    A = bundle_adjustment_sparsity(n_cams, n_points, camera_indices, point_indices)

    print("  Solving Least Squares...")
    res = least_squares(
        reprojection_residual, x0, 
        jac_sparsity=A, 
        verbose=2, 
        x_scale='jac', 
        ftol=1e-4, 
        xtol=1e-4,
        method='trf', 
        args=(n_cams, n_points, camera_indices, point_indices, points_2d, K)
    )
    
    optimized_params = res.x
    camera_params_opt = optimized_params[:n_cams*6].reshape((n_cams, 6))
    points_3d_opt = optimized_params[n_cams*6:].reshape((n_points, 3))
    
    cameras_opt = []
    for cp in camera_params_opt:
        rvec = cp[:3].reshape(3, 1)
        tvec = cp[3:6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        cameras_opt.append((R, tvec))
    
    return cameras_opt, points_3d_opt