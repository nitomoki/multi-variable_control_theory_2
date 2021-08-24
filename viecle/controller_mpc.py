import numpy as np
from curvature import *
import cvxopt
from cvxopt import matrix

def controller_mpc(Ahat, Bhat, What, Qhat, Rhat, n, m, N, hd, umax, curvature_map, V, t, x):
    """
    contoller_mpc

    Parameters
    ----------
    Ahat :
    Bhat :
    What :
    Qhat :
    Rhat :
    n :
    m :
    N :
    hd :
    umax :
    curvature_map :
    V :
    t :
    x :

    Returns
    -------
    u :
    """
    rhat = np.zeros((N,1))
    for k in range(N):
        rhat[k] = curvature(V, curvature_map, t+k*hd)

    Nn = N*n
    Nm = N*m
    
    P1 = np.hstack((Qhat, np.zeros((Nn,Nm)) ))
    P2 = np.hstack((np.zeros((Nm, Nn)), Rhat))
    P = np.vstack((P1, P2))
    P = matrix((P+P.T)/2)
    q = matrix(np.zeros((Nn+Nm, 1)))
    A = matrix(np.hstack((np.eye(Nn), -Bhat)))
    b = matrix(Ahat@x + What@rhat)
    G = matrix(np.array([]))
    h = matrix(np.array([]))
    
    sol = cvxopt.solvers.qp(P, q, A=A, b=b)

    return sol["x"][Nn,0]










