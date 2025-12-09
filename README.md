## Project Overview

1. Reconstruction: This project implements an Incremental Structure from Motion (SfM) pipeline in Python. It takes a sequence of 2D images of a bricked wall and reconstructs a sparse 3D point cloud of the scene.
The pipeline utilizes SIFT features for matching, solves the PnP problem to register new cameras, performs triangulation to generate 3D points and optimizes the reconstruction using Bundle Adjustment.
2. Visualization: An interactive, web-based Virtual Tour using Three.js that renders the reconstructed scene with smooth navigation mechanics.

## Phase 1: Structure from Motion (Python)
The reconstruction pipeline takes a sequence of 2D images of a bricked wall and generates a sparse 3D point cloud.

## Key Features

1. Feature Extraction: SIFT is used to detect keypoints and descriptors.
2. Geometric Verification: Matches are filtered using Homography (for planar surfaces) and Fundamental Matrix (for non-planar) checks.
3. Initialization: The reconstruction is seeded using an initial pair of frames (Frame 0 and Frame 2).
4. Incremental Reconstruction: New frames are added one by one via PnP and triangulation.
5. Optimization: Bundle Adjustment is applied periodically to minimize reprojection error.

## Code Description

1. main.py: The core pipeline. It handles the initialization using the Essential Matrix, loops through the dataset to register new frames using PnP, manages the 3D point cloud growth via triangulation and triggers Bundle Adjustment.
2. dataset_loader.py: Loads the image dataset. It implements a natural sort algorithm to ensure that frames are processed in the correct numerical order.
3. feature_utils.py: Implements detect_and_compute using OpenCV's SIFT. It extracts keypoints and descriptors from the input images.
4. matching_utils.py: Handles feature matching using BFMatcher. It applies Lowe’s Ratio Test to ensure only high-quality feature matches are kept.
5. pnp_utils.py: Contains estimate_camera_pose. This uses cv2.solvePnPRansac to determine the position of a new camera relative to the existing 3D point cloud.
6. triangulation_utils.py: Converts matched 2D points from two camera views into a 3D point using cv2.triangulatePoints. It includes checks to remove points behind the camera or too far away.
7. bundle_adjustment.py: Refines the camera poses and 3D point positions. It constructs a sparse Jacobian matrix and uses scipy.optimize.least_squares to minimize the reprojection error.
8. remove_outliers.py: A post-processing script. The raw output from the pipeline (reconstruction_result.ply) contains floating noise. This script filters points based on coordinate bounds and statistical distances, saving the cleaned result as reconstruction_clean.ply.
9. plot_3d.py: A visualization utility using Matplotlib. It plots the 3D point cloud with RGB colors and includes an internal filter to hide extreme statistical outliers during the interactive display.

## Results

 3D plot made using matplotlib
 ![alt text](3D_reconstruction.png)

 Visualisation on CloudCompare
 ![alt text](point_cloud_ss.bmp)

 ## Installation and Usage

1. Install Dependencies: pip install numpy opencv-python scipy matplotlib
2. Run the Reconstruction: python main.py
   
Output: reconstruction_result.ply and a 3D plot of the cloud

3. Clean Outliers: python remove_outliers.py
   
Output: reconstruction_clean.ply

## Viewing the Point Cloud

1. Specific Binary File: A file named point_cloud.bin is included in this repository. It can be opened directly in CloudCompare to rotate, zoom and inspect the structure in detail.

   OR
   
2. PLY Files: You can also import the generated reconstruction_clean.ply.


## Phase 2: Interactive Virtual Tour (Web/Three.js)
The second phase bridges the gap between raw data and user experience. It renders a high-density point cloud in a web browser, allowing users to navigate through the room via a curated path.

## Key Features
1. Coordinate Alignment: Automatically transforms the raw Photogrammetry data (Z-Up) to the WebGL coordinate system (Y-Up) by applying a -90° rotation to the world group.
2. Manual Node Curation: Uses a custom "Builder Tool" to define geometrically valid viewpoints, avoiding collisions with walls or ceilings.
3. Smooth Interpolation: Implements Linear Interpolation (Lerp) for position and Spherical Linear Interpolation (Slerp) for rotation to ensure cinematic camera movement.
4. Visual Polish: Features a clean UI, CSS-based cross-fading during transitions, and optimized binary PLY loading for performance. 

## Code Description

1. virtualtour.html: The final application. It loads the point cloud, renders the navigation nodes (blue spheres), and handles the user interaction logic.
2. index.html: A developer tool used to create the tour. It allows "Free Fly" navigation (WASD) to explore the scene and record coordinates for the final tour.
3. optimizer.py: A Python script utilizing the Open3D library. It downsamples the raw, high-density point cloud and converts it into a binary PLY format (room_binary.ply) to ensure low-latency loading in the web browser.

Usage
Start Local Server: Download the extension "Live Server" and open the html file "virtualtour.html" with live server extension. 

Controls:

Left Click: Rotate View

Scroll: Zoom In/Out

Click Blue Sphere: Fly to that location
