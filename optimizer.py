import open3d as o3d
import os

# CONFIGURATION
INPUT_FILE = "room.ply"
OUTPUT_FILE = "room_binary.ply"

def optimize_ply():

    print(f"Loading {INPUT_FILE}... (This might take a moment)")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    
    print(f"Points loaded: {len(pcd.points)}")
    
    # Downsample if you have more than 2 million points
    # This drastically speeds up loading without losing much visual quality
    if len(pcd.points) > 2000000:
        print("Downsampling point cloud to improve web performance...")
        pcd = pcd.voxel_down_sample(voxel_size=0.02) 
        print(f"New point count: {len(pcd.points)}")

    print(f"Saving optimized binary to {OUTPUT_FILE}...")
    o3d.io.write_point_cloud(OUTPUT_FILE, pcd, write_ascii=False)
    

if __name__ == "__main__":
    optimize_ply()