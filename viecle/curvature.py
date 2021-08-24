import numpy as np

def curvature(V, curvature_map, t):
    """
    curvature_map
    
    Parameters
    ----------
    V : double
    curvature_map : numpy.ndarray 2xNumber_of_curve
    t : double

    Returns
    -------
    rho : double
    """

    d = V*t

    for i in range(curvature_map.shape[1]):
        if d > curvature_map[0,i]:
            d = d - curvature_map[0,i]
        else:
            c = curvature_map[1,i]
            if c == 0.:
                return 0
            else:
                return 1./c

    return 0
