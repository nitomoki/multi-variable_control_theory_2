import numpy as np
from control import *
from vehicle import *
from curvature import *
from controller_mpc import *
from controller_lqr import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns





def main():
    sns.set()

    # input demension
    n = 5
    m = 1

    # physical parameters
    M = 350.
    I = 200.
    V = 20.*(1000./3600.)
    #V = 6.8*(1000./3600.)
    #V = 15.*(1000./3600.)
    lf = 1.
    lr = 0.28
    Kf = 10000
    Kr = 10000
    #umax = 0.1
    umax = 0.1
    xmax = [1., 0.5, 2., 1., 0.5]

    curvature_map = np.array([
            [35., 15.7, 31.4, 15.7, 35., 62.8, 22.6, 11.8, 23.6, 23.6, 23.6, 11.8, 22.6, 62.8],
            [0.,  -30., 30., -30.,  0., -20.,  0.,  -15.,  15., -15.,  15., -15.,  0.,  -20.]
            ])
    #curvature_map = np.array([
    #        [35., 15.7, 31.4, 15.7, 35., 62.8, 22.6, 11.8, 23.6, 23.6, 23.6, 11.8, 22.6, 62.8],
    #        [0.,  -30., 30., -30.,  0., -20.,  0.,  -20.,  20., -20.,  20., -20.,  0.,  -20.]
    #        ])

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
    R = 2000.0
    S = np.diag([100., 100., 100., 100., 100.])
    # feedback gain by lqr
    K, P, e = lqr(A, B, Q, R)

    # mpc
    N = 20
    hd = 0.05
    Ad = np.eye(n) + A*hd
    Bd = B*hd
    Wd = W*hd
    Qd = Q*hd
    Rd = R*hd
    Ahat = np.zeros((n*N, n))
    Bhat = np.zeros((n*N, m*N))
    What = np.zeros((n*N, m*N))
    Qhat = np.zeros((n*N, n*N))
    Rhat = np.zeros((m*N, m*N))

    for k in range(N):
        Ahat[k*n:(k+1)*n, 0:n] = np.linalg.matrix_power(Ad, k)

    for k1 in range(N):
        for k2 in range(k1, N):
            Bhat[k2*n:(k2+1)*n, k1*m:(k1+1)*m] = np.linalg.matrix_power(Ad, k2-k1)@Bd
            What[k2*n:(k2+1)*n, k1*m:(k1+1)*m] = np.linalg.matrix_power(Ad, k2-k1)@Wd

    for k in range(N):
        if not k == N-1:
            Qhat[k*n:(k+1)*n, k*n:(k+1)*n] = Qd
        else:
            Qhat[k*n:(k+1)*n, k*n:(k+1)*n] = S

        Rhat[k*m:(k+1)*m, k*m:(k+1)*m] = Rd

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

    while d < df:
        T = np.concatenate((T,np.array([t])))
        X = np.concatenate((X,x), axis=1)
        U = np.concatenate((U, np.array([u])))
        D = np.concatenate((D, np.array([d])))
        RHO = np.concatenate((RHO, np.array([rho])))
        #u = controller_mpc(Ahat, Bhat, What, Qhat, Rhat, n, m, N, hd, umax, curvature_map, V, t,x)
        u = controller_lqr(K, x)
        u = min(max(-umax,u),umax)
        xd = vehicle(A, B, W, V, curvature_map, t, x, u)

        rho = curvature(V, curvature_map, t)
        x = x + xd*dt
        d = d + V*dt
        t = t + dt
        bar.update(1)

    
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(611)
    ax1.plot(T, X[0,:], label='x1')
    ax1.set_xlabel('t')
    ax1.axes.xaxis.set_ticklabels([])
    ax1.legend(bbox_to_anchor=(0, 1), loc='upper left')

    ax2 = fig1.add_subplot(612)
    ax2.plot(T, X[1,:], label='x2')
    ax2.set_xlabel('t')
    ax2.axes.xaxis.set_ticklabels([])
    ax2.legend(bbox_to_anchor=(0, 1), loc='upper left')

    ax3 = fig1.add_subplot(613)
    ax3.plot(T, X[2,:], label='x3')
    ax3.set_xlabel('t')
    ax3.axes.xaxis.set_ticklabels([])
    ax3.legend(bbox_to_anchor=(0, 1), loc='upper left')

    ax4 = fig1.add_subplot(614)
    ax4.plot(T, X[3,:], label='x4')
    ax4.set_xlabel('t')
    ax4.axes.xaxis.set_ticklabels([])
    ax4.legend(bbox_to_anchor=(0, 1), loc='upper left')

    ax5 = fig1.add_subplot(615)
    ax5.plot(T, X[4,:], label='x5')
    ax5.set_xlabel('t')
    ax5.axes.xaxis.set_ticklabels([])
    ax5.legend(bbox_to_anchor=(0, 1), loc='upper left')

    axu = fig1.add_subplot(616)
    axu.plot(T, U, label='u')
    axu.set_xlabel('t')
    axu.legend(bbox_to_anchor=(0, 1), loc='upper left')

    #axd = fig2.add_subplot(211)
    #axd.plot(T, D)
    #axd.set_xlabel('t')
    #axd.set_ylabel('d')

    axrho = fig2.add_subplot(111)
    axrho.plot(T, RHO)
    axrho.set_xlabel('t')
    axrho.set_ylabel('ρ')

    mng = plt.get_current_fig_manager()
    plt.show()
    #u = controller_mpc(Ahat, Bhat, What, Qhat, Rhat, n, m, N, hd, umax, curvature_map, V, t,x)


if __name__ == "__main__":
    main()
