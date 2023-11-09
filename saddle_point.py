import numpy as np
from numpy.linalg import inv, lstsq

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
