import numpy as np

def controller_lqr(K, x):
    u = -K@x
    return u[0,0]
