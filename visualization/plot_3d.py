import matplotlib.pyplot as plt
import numpy as np

def plot_point_cloud(points_3d, colors=None):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    #  removing outliers
    if len(points_3d) > 10:
        center = np.median(points_3d, axis=0)
        # distance of every point from the center
        distances = np.linalg.norm(points_3d - center, axis=1)
        
        # within the 90th percentile of distance
        limit = np.percentile(distances, 90)
        mask = distances < limit
        
        points_vis = points_3d[mask]
        if colors is not None:
            colors_vis = colors[mask]
        else:
            colors_vis = None
            
        print(f"Visualizing {len(points_vis)} points (Removed {len(points_3d) - len(points_vis)} outliers)")
    else:
        points_vis = points_3d
        colors_vis = colors

    x = points_vis[:, 0]
    y = points_vis[:, 1]
    z = points_vis[:, 2]
    
    if colors_vis is not None:
        c = colors_vis / 255.0
        c = np.clip(c, 0.0, 1.0)
        ax.scatter(x, y, z, c=c, s=5, alpha=1.0, depthshade=False)
    else:
        ax.scatter(x, y, z, s=5, depthshade=False)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction ({len(points_vis)} points)')
    
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.invert_zaxis() 
    
    plt.show()