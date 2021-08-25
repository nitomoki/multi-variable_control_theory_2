import numpy as np
from control import *
from vehicle import *
from curvature import *
from controller_mpc import *
from controller_lqr import *
from plot import *
from map_creator import *
from tqdm import tqdm
import time

def main():

    # input demension
    n = 5
    m = 1

    # physical parameters
    M = 350.
    I = 200.
    V = 20.*(1000./3600.)
    lf = 1.
    lr = 0.28
    Kf = 10000
    Kr = 10000
    umax = 0.1

    course = 2
    curvature_map = map_creator(course)

    a11 = 2*(Kf + Kr)/M
    a12 = 2*(lf*Kf - lr*Kr)/M
    a21 = 2*(lf*Kf - lr*Kr)/I
    a22 = 2*(lf*lf*Kr + lr*lr*Kr)/I
    b1  = 2*Kf/M
    b2  = 2*lf*Kf/I

    A = np.array([
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., a11, -a11/V, -a12/V, b1],
        [0., a21, -a21/V, -a22/V, b2],
        [0., 0., 0., 0., 0.]
        ])

    B = np.array([
        [0],
        [0],
        [0],
        [0],
        [1]
        ])

    W = np.array([
        [0],
        [0],
        [-(a12 + V*V)],
        [-a22],
        [0]
        ])

    Mc = np.concatenate((B, A@B, A@A@B, A@A@A@B), axis=1)


    ### cost function setup
    # lqr
    Q = np.diag([1, 0.01, 0.01, 0.01, 0.01])
    R = 5000.0
    # feedback gain by lqr
    K, P, e = lqr(A, B, Q, R)

    # simulation
    x0 = np.array([[0], [0], [0], [0], [0]])
    dt = 0.01
    t = 0.
    d = 0.
    df = np.sum(curvature_map[0])
    x = x0
    u = 0
    T = []
    X = [[], [], [], [], []]
    U = []
    D = []
    RHO = []
    rho = 0.
    track = []

    bar = tqdm(total = (int)(df/(V*dt)))

    start = time.time()
    while d < df:
        T = np.concatenate((T,np.array([t])))
        X = np.concatenate((X,x), axis=1)
        U = np.concatenate((U, np.array([u])))
        D = np.concatenate((D, np.array([d])))
        RHO = np.concatenate((RHO, np.array([rho])))
        u = controller_lqr(K, x)
        u = min(max(-umax,u),umax)
        xd = vehicle(A, B, W, V, curvature_map, t, x, u)

        rho = curvature(V, curvature_map, t)
        x = x + xd*dt
        d = d + V*dt
        t = t + dt
        bar.update(1)

    max_error = np.amax(np.abs(X), axis=1)
    elapsed_time = time.time() - start
    title = 'Course ' + str(course) + '  LQR Controller   R:' + str(R) + "\nProcessing time: {0}".format(round(elapsed_time, 2)) + "[sec]" + "\nMax tracking error: {0}".format(round(max_error[0], 2)) + "[m]"
    plot(title, T, X, U, RHO, course=course)



if __name__ == "__main__":
    main()
