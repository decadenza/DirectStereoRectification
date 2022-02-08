#!/usr/bin/env python3
"""
Example of Direct Stereo Rectification Minimizing Perspective Distortion.
    
Copyright (C) 2020  Pasquale Lafiosca and Marta Ceccaroni

Modified by Kevin Cain (kevin@insightdigital.org) to provide calibration
from chessboard phtos via the canonical OpenCV methods.

Please note that the code below is based in part on the OpenCV documentation:
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

Excerpts are also adapted from Kaustubh Sadekar's open source calibration code:
https://github.com/spmallick/learnopencv/blob/master/stereo-camera/calibrate.py

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distbuted in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import numpy as np
from tqdm import tqdm

import rectification

if __name__ == "__main__":
    print("Direct Rectification EXAMPLE\n")

# Image path for calibration stereo pairs
pathL = "./chessboard_images/left/"
pathR = "./chessboard_images/right/"

print("Extract (x, y) coordinates for calibration chessboard\n")

# Termination criteria for refining the detected corners on 9x6 corner grid
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

img_ptsL = []
img_ptsR = []
obj_pts = []

# We assume (31) total chessboard stereo pairs for calibration
for i in tqdm(range(1,31)):
	imgL = cv2.imread(pathL+"%d_1.png"%i)
	imgR = cv2.imread(pathR+"%d_0.png"%i)
	imgL_gray = cv2.imread(pathL+"%d_1.png"%i,0)
	imgR_gray = cv2.imread(pathR+"%d_0.png"%i,0)

	outputL = imgL.copy()
	outputR = imgR.copy()

	retR, cornersR =  cv2.findChessboardCorners(outputR,(9,6),None)
	retL, cornersL = cv2.findChessboardCorners(outputL,(9,6),None)

	if retR and retL:
		obj_pts.append(objp)
		cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
		cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
		cv2.drawChessboardCorners(outputR,(9,6),cornersR,retR)
		cv2.drawChessboardCorners(outputL,(9,6),cornersL,retL)

		# Display corners (sanity check for user)
		cv2.imshow('cornersR',outputR)
		cv2.imshow('cornersL',outputL)
		cv2.waitKey(0)

		img_ptsL.append(cornersL)
		img_ptsR.append(cornersR)

print("Calculating left camera parameters ... ")
# Calibrating left camera
retL, mtxL, distCoeffs1, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
hL,wL= imgL_gray.shape[:2]
new_mtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distCoeffs1,(wL,hL),1,(wL,hL))

print("Calculating right camera parameters ... ")
# Calibrating right camera
retR, mtxR, distCoeffs2, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
hR,wR= imgR_gray.shape[:2]
new_mtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distCoeffs2,(wR,hR),1,(wR,hR))


print("Stereo calibration .....")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Lock the intrinsic camara matrixes; calculate only rotation, translation, Fundamental and Essential matrices
# Intrinsic parameters for each camera remain unchanged

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Estimate transformation of camera two relative to camera one, and produce the Essential and Fundamental matrices
retS, new_mtxL, distCoeffs1, new_mtxR, distCoeffs2, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, new_mtxL, distCoeffs1, new_mtxR, distCoeffs2, imgL_gray.shape[::-1], criteria_stereo, flags)

# Load images for rectification
img1 = cv2.imread("img/left.png")           # Left image
img2 = cv2.imread("img/right.png")          # Right image
dims1 = img1.shape[::-1][1:]                # Image dimensions as (width, height)
dims2 = img2.shape[::-1][1:]

# Calibration data
# Note: Principal Points arbitrarily set at image center
A1 = np.array([[ 640, 0, 640/2], [0, 640, 480/2], [0,0,1]])             # Left camera intrinsic matrix
A2 = np.array([[ 640, 0, 640/2], [0, 640, 480/2], [0,0,1]])             # Right camera intrinsic matrix
RT1 = np.array([[ 1, 0, 0,  0],   # Left extrinsic parameters
                [0, 1, 0, 0],
                [0, 0, 1, 0]])
# Right extrinsic parameters
RT2 = np.hstack((Rot, Trns.reshape(3,1) ))

print("RT2: ", RT2)
print("distCoeffs1: ", distCoeffs1)
print("distCoeffs2: ", distCoeffs2)

# Distortion coefficients
# Lens distortion calculated per camera by cv2.calibrateCamera
# For synthetic images without lens distortion, use empty arrays for the distortion coefficients:
#distCoeffs1 = np.array([])
#distCoeffs2 = np.array([])

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
