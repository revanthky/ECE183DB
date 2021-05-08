""" Tools for synthesising controllers for LTI systems.
(c) 2014 Mark W. Mueller
"""

import numpy as np
from scipy import linalg as la

from controlpy import analysis

def controller_lqr(A, B, Q, R):
    """Solve the continuous time LQR controller for a continuous time system.
    
    A and B are system matrices, describing the systems dynamics:
     dx/dt = A x + B u
    
    The controller minimizes the infinite horizon quadratic cost function:
     cost = integral (x.T*Q*x + u.T*R*u) dt
    
    where Q is a positive semidefinite matrix, and R is positive definite matrix.
    
    Returns K, X, eigVals:
    Returns gain the optimal gain K, the solution matrix X, and the closed loop system eigenvalues.
    The optimal input is then computed as:
     input: u = -K*x
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    #compute the LQR gain
    K = np.dot(np.linalg.inv(R),(np.dot(B.T,X)))
    
    eigVals = np.linalg.eigvals(A-np.dot(B,K))
    
    return K, X, eigVals

A = np.array([  [1.0,  0,   0],[  0,1.0,   0],[  0,  0, 1.0]])
B = np.array([[np.cos(beta)*np.cos(alpha)*deltat, 0, 0, 0],
                  [np.cos(beta)*np.sin(alpha)*deltat, 0, 0, 0],
                  [np.sin(beta)*deltat, 0, 0, 0],
                  [0, deltat, 0, 0],
                  [0, 0, deltat, 0],
                  [0, 0, 0, deltat]])
Q=
R=
controller_lqr()