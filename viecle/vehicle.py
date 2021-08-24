import numpy as np
from curvature import *

def vehicle(A, B, W, V, curvature_map, t, x, u):
    """
    function: vehicle

    Parameters
    ----------
    A : numpy.ndarray 5x5
    B : numpy.ndarray 5x1
    W : numpy.ndarray 5x1
    V : double
    curvature_map : numpy.ndarray 2xNumber_of_curve
    t : double
    x : numpy.ndarray 5x1
    u : double

    Returns
    -------
    xd : numpy.ndarray 5x1
    """

    rho = curvature(V, curvature_map, t)
    xd = A@x + B*u + W*rho
    #xd = A@x  + W*rho
    return xd

