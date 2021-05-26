#!/usr/bin/env python3
"""
Example of Direct Stereo Rectification Minimizing Perspective Distortion.
    
Copyright (C) 2020  Pasquale Lafiosca and Marta Ceccaroni

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import numpy as np

import rectification

if __name__ == "__main__":
    print("Direct Rectification EXAMPLE")

    # Load images
    img1 = cv2.imread("img/left.png")           # Left image
    img2 = cv2.imread("img/right.png")          # Right image
    dims1 = img1.shape[::-1][1:]                # Image dimensions as (width, height)
    dims2 = img2.shape[::-1][1:]
    
    # Calibration data
    A1 = np.array([[ 960, 0, 960/2], [0, 960, 540/2], [0,0,1]])             # Left camera intrinsic matrix
    A2 = np.array([[ 960, 0, 960/2], [0, 960, 540/2], [0,0,1]])             # Right camera intrinsic matrix
    RT1 = np.array([[ 0.98920029, -0.11784191, -0.08715574,  2.26296163],   # Left extrinsic parameters
                    [-0.1284277 , -0.41030705, -0.90285909,  0.15825593],
                    [ 0.07063401,  0.90430164, -0.42101002, 11.0683527 ]])
    RT2 = np.array([[ 0.94090474,  0.33686835,  0.03489951,  1.0174818 ],   # Right extrinsic parameters
                    [ 0.14616159, -0.31095025, -0.93912017,  2.36511779],
                    [-0.30550784,  0.88872361, -0.34181178, 14.08488464]])

    # Distortion coefficients
    # Empty because we're using digitally acquired images (no lens distortion).
    # See OpenCV distortion parameters for help.
    distCoeffs1 = np.array([])   
    distCoeffs2 = np.array([])
    
    # 3x4 camera projection matrices
    Po1 = A1.dot(RT1)                               
    Po2 = A2.dot(RT2)
    
    # Fundamental matrix F is usually known from calibration, alternatively
    # you can get the Fundamental matrix from projection matrices
    F = rectification.getFundamentalMatrixFromProjections(Po1, Po2)
    
    # ANALYTICAL RECTIFICATION to get the **rectification homographies that minimize distortion**
    # See function dr.getAnalyticalRectifications() for details
    Rectify1, Rectify2 = rectification.getDirectRectifications(A1, A2, RT1, RT2, dims1, dims2, F)
    
    # Final rectified image dimensions (common to both images)
    destDims = dims1
    
    # Get fitting affine transformation to fit the images into the frame
    # Affine transformations do not introduce perspective distortion
    Fit1, Fit2 = rectification.getFittingMatrices(Rectify1, Rectify2, dims1, dims2, destDims=dims1)
    
    # Compute maps with OpenCV considering rectifications, fitting transformations and lens distortion
    # These maps can be stored and applied to rectify any image pair of the same stereo rig
    mapx1, mapy1 = cv2.initUndistortRectifyMap(A1, distCoeffs1, Rectify1.dot(A1), Fit1, destDims, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(A2, distCoeffs2, Rectify2.dot(A2), Fit2, destDims, cv2.CV_32FC1)
    
    # Apply final transformation to images 
    img1_rect = cv2.remap(img1, mapx1, mapy1, interpolation=cv2.INTER_LINEAR);
    img2_rect = cv2.remap(img2, mapx2, mapy2, interpolation=cv2.INTER_LINEAR);
    
    # Draw a line as reference (optional)
    img1_rect = cv2.line(img1_rect, (0,int((destDims[1]-1)/2)), (destDims[0]-1,int((destDims[1]-1)/2)), color=(0,0,255), thickness=1)
    img2_rect = cv2.line(img2_rect, (0,int((destDims[1]-1)/2)), (destDims[0]-1,int((destDims[1]-1)/2)), color=(0,0,255), thickness=1)
    
    # Print some info
    perspDist = rectification.getLoopZhangDistortionValue(Rectify1, dims1)+rectification.getLoopZhangDistortionValue(Rectify2, dims2)
    print("Perspective distortion:", perspDist)
    
    # Show images
    cv2.imshow('LEFT Rectified', img1_rect)
    cv2.imshow('RIGHT Rectified', img2_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
