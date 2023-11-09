import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Covert 4x4 homogeneous pose matrix to x, y, z, roll, pitch, yaw."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Covert x, y, z, roll, pitch, yaw to 4x4 homogeneous pose matrix."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T

def pose_estimate_nls(K, Twc_guess, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    K          - 3x3 camera intrinsic calibration matrix.
    Twc_guess  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts       - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts       - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array (float64), pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---

    # Some hints on structure are included below...

    # 1. Convert initial guess to parameter vector (6 x 1).
    # ...
    
    E=epose_from_hpose(inv(Twc_guess))

    iter = 1

    # 2. Main loop - continue until convergence or maxIters.
    while True:
        # 3. Save previous best pose estimate.
        # ...
        
        Twc = inv(hpose_from_epose(E))

        # 4. Project each landmark into image, given current pose estimate.
        
        jjt = np.zeros(36).reshape((6,6))
        jte = np.zeros(6).reshape((6,1))
        for i in np.arange(tp):
            wpt = np.vstack((np.array([Wpts[0][i]]), np.array([Wpts[1][i]]), np.array([Wpts[2][i]])))
            twpt = np.vstack((np.array([Wpts[0][i]]), np.array([Wpts[1][i]]), np.array([Wpts[2][i]]), np.array([1])))
            ipt = np.vstack((np.array([Ipts[0][i]]), np.array([Ipts[1][i]])))
            J = find_jacobian(K,Twc,wpt)
            
            tk = np.hstack((K,np.array([[0],[0],[0]])))
            tk = np.vstack((tk,np.array([0,0,0,1])))
            tx = tk@inv(Twc)@twpt
            x = tx[0:2]/tx[2]
            e = x - ipt

            jjt = jjt + J.T@J
            jte = jte + J.T@e

        # 5. Solve system of normal equations for this iteration.
        # ...

        new = inv(jjt)@jte
        E = epose_from_hpose(inv(Twc))
        prev_param = E.copy()
        for i in range(0,6):
            E[i][0] = E[i][0] + new[i][0]
        
        param = E.copy()
        # 6. Check - converged?
        diff = norm(param - prev_param)

        if norm(diff) < 1e-13:
            print("Covergence required %d iters." % iter)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iter)
            break
        
        iter += 1

    # 7. Compute and return homogeneous pose matrix Twc.
    Twc = inv(hpose_from_epose(E))
    #------------------

    correct = isinstance(Twc, np.ndarray) and \
        Twc.dtype == np.float64 and \
        Twc.shape == (4, 4) and Twc[3, 3] == 1.0000

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Twc
