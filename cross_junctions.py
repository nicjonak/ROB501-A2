import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.


def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    # Code goes here.

    #Find image size x,y,total
    xpx = I.shape[0]
    ypx = I.shape[1]
    totpx = xpx*ypx
    offset = 0.5
    
    #Create matrices to represent equation (4) 
    a = np.zeros((totpx,6))
    b = np.zeros((totpx,1))
    for x in range(xpx):
        for y in range(ypx):
            xo = ((x+0.5)/xpx) - offset
            yo = ((y+0.5)/ypx) - offset
            arr = np.array((xo**2, xo*yo, yo**2,xo,yo,1))
            a[ypx*x+y] = arr
            b[ypx*x+y] = I[x][y]

    #Find desired variables
    variables = lstsq(a,b)
    alpha = variables[0][0]
    beta = variables[0][1]
    gamma = variables[0][2]
    delta = variables[0][3]
    epsilon = variables[0][4]
    
    #Compute saddle point
    A = -inv(np.array([[2*alpha,beta],[beta,2*gamma]]).reshape((2,2)))
    B = np.array([[delta],[epsilon]]).reshape((2,1))

    pt = np.matmul(A,B)
    xt = pt[0]*xpx + xpx//2
    yt = pt[1]*ypx + ypx//2
    pt[1] = xt-0.5
    pt[0] = yt-0.5
    

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt


def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---

    #------------------
    At=[]
    for i in range(0,4):
      xi=I1pts[0][i]
      yi=I1pts[1][i]
      ui=I2pts[0][i]
      vi=I2pts[1][i]
      ai=np.array([[-xi,-yi,-1,0,0,0,ui*xi,ui*yi,ui],
          [0,0,0,-xi,-yi,-1,vi*xi,vi*yi,vi]])
      At=At+[ai]
    AU=np.concatenate((At[0],At[1]),axis=0)
    AL=np.concatenate((At[2],At[3]),axis=0)
    A=np.concatenate((AU,AL),axis=0)
    ht=null_space(A)
    H=np.array([[ht[0][0],ht[1][0],ht[2][0]],
                [ht[3][0],ht[4][0],ht[5][0]],
                [ht[6][0],ht[7][0],ht[8][0]]])
    H=H*(1/H[2][2]) 
    return H, A


def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---

    # Code goes here...
    
    nc = Wpts.shape[1]
    #Ipts = np.zeros((2, nc))
    
    #Find bounding polygon in world coordinates
    cnt=0
    for i in range(1,nc):   #Find how many cross junctions in each row
        if Wpts[0][i] !=0:
            cnt=cnt+1
        else:
            break
    
    size=63.5/1000
    dxToBorder = (1/3)*size+size #Distances from corner cross junctions to borders of bounding polygon
    dyToBorder = (1/5)*size+size

    uplw = (Wpts[:,0] + np.array([-dxToBorder, -dyToBorder, 1])).reshape(3,1)
    uprw = (Wpts[:,cnt] + np.array([dxToBorder, -dyToBorder, 1])).reshape(3,1)
    botlw = (Wpts[:,nc-cnt-1] + np.array([-dxToBorder, dyToBorder, 1])).reshape(3,1)
    botrw = (Wpts[:,nc-1] + np.array([dxToBorder, dyToBorder, 1])).reshape(3,1)
    wcorners = np.concatenate((uplw, uprw, botrw, botlw), axis=1) #Bounding polygon in world coords

    #Homography between world and image
    H, A = dlt_homography(wcorners, bpoly)
    
    #Homographied junction coordinates (the given world points in image coords)
    nwpts = Wpts.copy()
    nwpts[2,:] = 1 #New world points with z=1
    hpts = np.matmul(H,nwpts)
    hpts = (hpts/hpts[2,:])[:2,:]
    
    #Use saddle_point to find more precise junction coords for output
    out=[]
    window=10
    for x,y in np.transpose(hpts):
        x=int(x)
        y=int(y)
        sp = saddle_point(I[y-window:y+window,x-window:x+window])
        out.append([x+sp[0,0]-window,y+sp[1,0]-window])
    
    Ipts=np.transpose(np.array(out))
    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts
