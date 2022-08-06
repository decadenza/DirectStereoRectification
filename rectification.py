"""
Direct Stereo Rectification Minimizing Perspective Distortion.
    
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
import math
import numpy as np


    
def getFundamentalMatrixFromProjections(P1,P2):
    """
    Compute the fundamental matrix from a couple of 3x4 projection matrices.
    
    Parameters
    ----------
    P1,P2 : numpy.ndarray
        3x4 projection matrices.
    
    Returns
    -------
    numpy.ndarray
        The fundamental matrix.
    """
    X = []
    X.append(np.vstack((P1[1,:], P1[2,:])))
    X.append(np.vstack((P1[2,:], P1[0,:])))
    X.append(np.vstack((P1[0,:], P1[1,:])))

    Y = []
    Y.append(np.vstack((P2[1,:], P2[2,:])))
    Y.append(np.vstack((P2[2,:], P2[0,:])))
    Y.append(np.vstack((P2[0,:], P2[1,:])))

    F = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            F[i, j] = np.linalg.det(np.vstack((X[j], Y[i])))
    
    return F

   
def getMinYCoord(H, dims):
    """
    Get the minimum Y coordinate after a transformation H.
    
    Please refer to "Computying rectifying homographies for stereo vision", CVPR 1999, Loop C. and Zhang Z.
    
    Parameters
    ----------
    H : numpy.ndarray
        A 3x3 transformation.
    dims : tuple
        Image dimensions as (width,height).
    
    Returns
    -------
    float
        The minimum Y coordinate after the transformation.
    """
    tL = H.dot(np.array([[0],[0],[1]]))[:,0]
    tL = tL/tL[2]
    bL = H.dot(np.array([[0],[dims[1]-1],[1]]))[:,0]
    bL = bL/bL[2]
    tR = H.dot(np.array([[dims[0]-1],[0],[1]]))[:,0]
    tR = tR/tR[2]
    bR = H.dot(np.array([[dims[0]-1],[dims[1]-1],[1]]))[:,0]
    bR = bR/bR[2]
    
    return min(tL[1], tR[1], bR[1], bL[1])


def getShearingTransformation(H, dims):
    """
    Compute the optimal shearing transformation.
    
    Please refer to "Computying rectifying homographies for stereo vision", CVPR 1999, Loop C. and Zhang Z.
    
    Parameters
    ----------
    H : numpy.ndarray
        A 3x3 transformation.
    dims : tuple
        Image dimensions as (width,height).
    
    Returns
    -------
    numpy.ndarray
        A 3x3 X-shearing transformation.
    """
    a = H.dot([(dims[0]-1)/2, 0, 1])            # Top middlepoint
    b = H.dot([(dims[0]-1), (dims[1]-1)/2, 1])  # Right middlepoint
    c = H.dot([(dims[0]-1)/2, (dims[1]-1), 1])  # Bottom middlepoint
    d = H.dot([0, (dims[1]-1)/2, 1])            # Left middlepoint
    a = a / a[2]
    b = b / b[2]
    c = c / c[2]
    d = d / d[2]
    
    # Get lines
    x = b - d
    y = c - a
    
    # Calculate coefficients
    a_coeff = ( (dims[1]*x[1])**2 + (dims[0]*y[1])**2 ) / ( dims[0]*dims[1]*(x[1]*y[0] - x[0]*y[1]) )
    b_coeff = ( (dims[1]**2)*x[0]*x[1] + (dims[0]**2)*y[0]*y[1] ) / ( dims[0]*dims[1]*(x[0]*y[1] - x[1]*y[0]) )
    
    # Build shearing matrix transform
    S = np.array([[a_coeff,b_coeff,0],[0,1,0],[0,0,1]])
    
    return S


def getDirectRectifications(A1, A2, RT1, RT2, dims1, dims2, F):
    """
    Compute the analytical rectification homographies.
    
    Compute the 3x3 transformations to rectify a couple of stereo images
    with minimim perspective distortion.
    
    Parameters
    ----------
    A1, A2 : numpy.ndarray
        3x3 camera matrices of intrinsic parameters.
    RT1, RT2 : numpy.ndarray
        3x4 extrinsic parameters matrices.
    dims1, dims2: tuple
        Original images dimensions as (width, height).
    F : numpy.ndarray
        3x3 fundamental matrix.
    
    Returns
    -------
    Rectify1, Rectify2 : numpy.ndarray
        3x3 rectification homographies.
    """
    if np.all(np.equal(F/F[2,1], np.array([[0,0,0],[0,0,-1],[0,1,0]]))):
        # PARTICULAR CASE 1: Stereo rig is already rectified
        # No perspective transformation is needed
        w1 = w2 = np.array([0,0,1])
    
    else:
        # Baseline vector in world coord (cam1 -> cam2)
        bv = np.linalg.inv(RT2[:,:3]).dot(RT2[:,3]) - np.linalg.inv(RT1[:,:3]).dot(RT1[:,3])
        
        # Auxiliary matrices
        B = ( bv.dot(bv) * np.eye(3) - bv[:,None].dot(bv[None,:]) ).dot(np.linalg.inv(A1.dot(RT1[:,:3])))
        L1 = np.transpose(np.linalg.inv(A1.dot(RT1[:,:3]))).dot(B)
        L2 = np.transpose(np.linalg.inv(A2.dot(RT2[:,:3]))).dot(B)
        
        # Auxiliary matrices II (as in Loop-Zhang algorithm)
        # N.B. The variable P1 is actually P1.P1^T and so on.
        P1 = (dims1[0]*dims1[1]/12)*np.array([[dims1[0]**2 - 1, 0, 0],[0, dims1[1]**2 - 1,0],[0, 0, 0]])
        Pc1 = np.array([[(dims1[0] - 1)**2/4, (dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[0] - 1)/2], [(dims1[0] - 1)*(dims1[1] - 1)/4, (dims1[1] - 1)**2/4, (dims1[1] - 1)/2],[(dims1[0] - 1)/2, (dims1[1] - 1)/2, 1]])
        P2 = (dims2[0]*dims2[1]/12)*np.array([[dims2[0]**2 - 1, 0, 0],[0, dims2[1]**2 - 1,0],[0, 0, 0]])
        Pc2 = np.array([[(dims2[0] - 1)**2/4, (dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[0] - 1)/2], [(dims2[0] - 1)*(dims2[1] - 1)/4, (dims2[1] - 1)**2/4, (dims2[1] - 1)/2],[(dims2[0] - 1)/2, (dims2[1] - 1)/2, 1]])
        
        M1 = L1.T.dot(P1).dot(L1)
        C1 = L1.T.dot(Pc1).dot(L1)
        M2 = L2.T.dot(P2).dot(L2)
        C2 = L2.T.dot(Pc2).dot(L2)
        
        # Polynomial coefficients
        m1 = M1[1,2]*C1[1,2] - M1[2,2]*C1[1,1]
        m2 = M1[1,1]*C1[1,2] - M1[1,2]*C1[1,1]
        
        if np.all(np.equal(RT1[:,:3], RT2[:,:3])) and np.all(np.equal(A1, A2)) and np.all(np.equal(P1, P2)) and np.all(np.equal(Pc1, Pc2)):
            # PARTICULAR CASE 2: The cameras have the same orientation: we have a single solution
            sol = [-m1/m2]
            
        else:
            
            # Polynomial coefficients II
            m3 = C2[1,2]/C2[1,1]
            m4 = C2[1,1]/C1[1,1]
            m5 = M2[1,2]*C2[1,2] - M2[2,2]*C2[1,1]
            m6 = M2[1,1]*C2[1,2] - M2[1,2]*C2[1,1]
            m7 = C1[1,2]/C1[1,1]
            m8 = 1/m4
            
            a = m2*m4 + m6*m8
            b = m1*m4 + 3*m2*m3*m4 + m5*m8 + 3*m6*m7*m8
            c = 3*(m1*m3*m4 + m2*m3**2*m4 + m5*m7*m8 + m6*m7**2*m8)
            d = 3*m1*m3**2*m4 + m2*m3**3*m4 + 3*m5*m7**2*m8 + m6*m7**3*m8
            e = m1*m3**3*m4 + m5*m7**3*m8
            
            # 4th degree equation formula
            p = (8*a*c - 3 * b**2 ) / (8 * a**2)
            q = 12*a*e - 3*b*d + c**2
            s = 27*a*d**2 - 72*a*c*e + 27*b**2*e - 9*b*c*d + 2*c**3
            D0 = math.pow( (1/2)*(s+math.sqrt(s**2 - 4*q**3)), 1/3)
            Q = (1/2) * math.sqrt( -(2/3)*p + 1/(3*a) * (D0 + q / D0) )
            S = ( 8*a**2*d - 4*a*b*c + b**3 ) / ( 8*a**3 ) 
            
            # Take acceptable solutions only
            sol = []
            if -4*Q**2 - 2*p + S/Q >= 0:
                sol.append( -b / (4*a) - Q - (1/2)*math.sqrt( -4*Q**2 - 2*p + S/Q) )
                sol.append( -b / (4*a) - Q + (1/2)*math.sqrt( -4*Q**2 - 2*p + S/Q) )
            
            if -4*Q**2 - 2*p - S/Q >= 0:
                sol.append( -b / (4*a) + Q - (1/2)*math.sqrt( -4*Q**2 - 2*p - S/Q) )
                sol.append( -b / (4*a) + Q + (1/2)*math.sqrt( -4*Q**2 - 2*p - S/Q) )
            
            if len(sol)<1:
                raise ValueError("No analitic solution.")
        
           
        def getW(ss):
            # Inner function to compute w1 and w2 from the solution
            
            # Point over image 1 in world coordinates
            p1w = np.linalg.inv(RT1[:,:3]).dot( np.linalg.inv(A1).dot(np.array([0,ss,1])) - RT1[:,3] )
            # New x axis
            xv = bv / np.linalg.norm(bv)
            # Projection on the baseline of the vector p1w - C2 in world coordinates
            oop1w = ( p1w + np.linalg.inv(RT2[:,:3]).dot(RT2[:,3]) ).dot(xv) * xv - np.linalg.inv(RT2[:,:3]).dot(RT2[:,3])
            
            zv = p1w - oop1w                # New z axis
            yv = np.cross(zv, bv)           # New y axis
            yv = yv / np.linalg.norm(yv)    # Normalize y direction
            zv = zv / np.linalg.norm(zv)    # Normalize z direction
            Rnew = np.array([xv,yv,zv])     # New camera orientation
            
            # Loop-Zhang w1 and w2
            w1 = Rnew.dot( np.linalg.inv(A1.dot(RT1[:,:3])) )[2,:]
            w2 = Rnew.dot( np.linalg.inv(A2.dot(RT2[:,:3])) )[2,:]
            w1 = w1 / w1[2]                 # Rescale with 3rd coordinate as 1
            w2 = w2 / w2[2]
            #l = -w1[1]/w1[0]               # Loop-Zhang lambda parameter (not needed)
            
            return w1, w2
        
        
        def getDistortion(s):
            # Inner function as compact version of getLoopZhangDistortionValue()
            w1, w2 = getW(s)    
            dist1 = float( w1.dot(P1).dot(w1)/w1.dot(Pc1).dot(w1) )
            dist2 = float( w2.dot(P2).dot(w2)/w2.dot(Pc2).dot(w2) )
            return dist1+dist2
            
        
        # Find minimum distortion among admissible solutions (4 or 2 solutions)
        bestSol = min(zip( sol, map(getDistortion, sol)), key=lambda x:x[1])[0]
        # Get associated w1 and w2
        w1, w2 = getW(bestSol)
    
    # At this point we have the correct w1 and w2
    # From here we follow the rest of the Loop-Zhang algorithm
        
    # Build projective transforms
    Hp1 = np.array([ [1,0,0], [0,1,0], w1 ])
    Hp2 = np.array([ [1,0,0], [0,1,0], w2 ])
    
    # Calculate vc2 so that "the minimum w-coordinate of a pixel in either image is zero."
    vc2 = -min( getMinYCoord(Hp1, dims1), getMinYCoord(Hp2, dims2) )
    
    # Build similarity transforms
    Hr1 = np.array([ [F[2,1]-w1[1]*F[2,2], w1[0]*F[2,2]-F[2,0], 0], \
                     [w1[0]*F[2,2]-F[2,0], w1[1]*F[2,2]-F[2,1], -(F[2,2] + vc2)], \
                     [0, 0, 1] ]) 
    
    Hr2 = np.array([ [F[1,2]-w2[1]*F[2,2], w2[0]*F[2,2]-F[0,2], 0], \
                     [F[0,2]-w2[0]*F[2,2], F[1,2]-w2[1]*F[2,2], vc2], \
                     [0, 0, 1] ])
    
    # Combine perspective and similarity transformations
    Hrp1 = Hr1.dot(Hp1)
    Hrp2 = Hr2.dot(Hp2)
    
    # Find best shearing transformations
    Hs1 = getShearingTransformation(Hrp1, dims1)
    Hs2 = getShearingTransformation(Hrp2, dims2)
    
    # Get final rectification transformations
    Rectify1 = Hs1.dot(Hrp1)
    Rectify2 = Hs2.dot(Hrp2)
    
    return Rectify1, Rectify2


def getLoopZhangDistortionValue(Hp, dims):
    """
    Return the perspective distortion value.
    
    Please refer to eq. (10) of "Computying rectifying homographies for stereo vision", CVPR 1999, Loop C. and Zhang Z.
    
    Parameters
    ----------
    Hp : numpy.ndarray
        3x3 transformation.
    dims : tuple
        Image dimensions as (width, height).
        
    Returns
    -------
    float
        Perspective distortion introduced by Hp on the image.
    """
    PPt = np.array( (dims[0]*dims[1]/12) * np.array([ [dims[0]**2 - 1, 0, 0],[0, dims[1]**2 - 1,0],[0, 0, 0] ]) )
    PcPct = np.array( np.array([ [(dims[0] - 1)**2/4, (dims[0] - 1)*(dims[1] - 1)/4, (dims[0] - 1)/2], [(dims[0] - 1)*(dims[1] - 1)/4, (dims[1] - 1)**2/4, (dims[1] - 1)/2],[(dims[0] - 1)/2, (dims[1] - 1)/2, 1] ]) )
    w = np.vstack(Hp[2,:])
    
    return float( w.T.dot(PPt).dot(w)/w.T.dot(PcPct).dot(w) )


def getFittingMatrices(H1, H2, dims1, dims2, destDims=None, zoom=1):
    """
    Compute affine tranformation to fit the rectified images into desidered dimensions.
    
    After rectification usually the image is no more into the original image bounds.
    One can apply any transformation that do not affect disparity to fit the image into boundaries.
    This function corrects flipped images too.
    The algorithm may fail if one epipole is too close to the image.
    
    Parameters
    ----------
    H1, H2 : numpy.ndarray
        3x3 rectifying homographies.
    dims1, dims2 : tuple
        Resolution of images as (width, height) tuple.
    distCoeffs1, distCoeffs2 : numpy.ndarray, optional
        Distortion coefficients in the order followed by OpenCV. If None is passed, zero distortion is assumed.
    destDims : tuple, optional
        Resolution of destination images as (width, height) tuple (default to the first image resolution).
    zoom : float, optional
        Zoom parameter to be applied to both images (default to 1). Used to remove unwanted portions of the images.
        
    Returns
    -------
    numpy.ndarray
        3x3 affine transformation to be used both for the first and for the second camera.
    """
    if not destDims:
        destDims=dims1
    
    # Calculate image 1 boundaries
    tL1 = H1.dot(np.array([[0],[0],[1]]))[:,0]
    tL1 = tL1/tL1[2]
    bL1 = H1.dot(np.array([[0],[dims1[1]-1],[1]]))[:,0]
    bL1 = bL1/bL1[2]
    tR1 = H1.dot(np.array([[dims1[0]-1],[0],[1]]))[:,0]
    tR1 = tR1/tR1[2]
    bR1 = H1.dot(np.array([[dims1[0]-1],[dims1[1]-1],[1]]))[:,0]
    bR1 = bR1/bR1[2]
    
    # Calculate image 2 boundaries
    tL2 = H2.dot(np.array([[0],[0],[1]]))[:,0]
    tL2 = tL2/tL2[2]
    bL2 = H2.dot(np.array([[0],[dims2[1]-1],[1]]))[:,0]
    bL2 = bL2/bL2[2]
    tR2 = H2.dot(np.array([[dims2[0]-1],[0],[1]]))[:,0]
    tR2 = tR2/tR2[2]
    bR2 = H2.dot(np.array([[dims2[0]-1],[dims2[1]-1],[1]]))[:,0]
    bR2 = bR2/bR2[2]
    
    minX1 = min(tR1[0], bR1[0], bL1[0], tL1[0])
    minX2 = min(tR2[0], bR2[0], bL2[0], tL2[0])
    maxX1 = max(tR1[0], bR1[0], bL1[0], tL1[0])
    maxX2 = max(tR2[0], bR2[0], bL2[0], tL2[0])
    
    minY = min(tR2[1], bR2[1], bL2[1], tL2[1], tR1[1], bR1[1], bL1[1], tL1[1])
    maxY = max(tR2[1], bR2[1], bL2[1], tL2[1], tR1[1], bR1[1], bL1[1], tL1[1])
    
    # Flip factor
    flipX = 1
    flipY = 1
    if tL1[0]>tR1[0]:
        flipX = -1
    if tL1[1]>bL1[1]:
        flipY = -1
    
    # Scale X (choose scale X to best fit bigger image between left and right)
    if(maxX2 - minX2 > maxX1 - minX1):
        scaleX = flipX * zoom * destDims[0]/(maxX2 - minX2)
    else:
        scaleX = flipX * zoom * destDims[0]/(maxX1 - minX1)
    
    # Scale Y (unique not to lose rectification) 
    scaleY = flipY * zoom * destDims[1]/(maxY - minY)
    
    # Translation X (keep always at left border)
    if flipX == 1:
        tX1 = -minX1 * scaleX
        tX2 = -minX2 * scaleX
    else:
        tX1 = -maxX1 * scaleX
        tX2 = -maxX2 * scaleX
    
    # Translation Y (keep always at top border)
    if flipY == 1:
        tY = -minY * scaleY
    else:
        tY = -maxY * scaleY
        
    K1 = np.array( [[scaleX,0,tX1], [0,scaleY,tY], [0,0,1]] )
    K2 = np.array( [[scaleX,0,tX2], [0,scaleY,tY], [0,0,1]] )
    
    return K1, K2

    
    

    
