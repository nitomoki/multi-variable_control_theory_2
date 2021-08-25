import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(title, T, X, U, RHO, show_course=False, plot_limit=True, course=1):
    sns.set()
    xmax = [[-10., 10.], [-1.3, 1.3], [-6., 6.], [-1.3, 1.3], [-0.5, 0.5]]
    umax = [-0.11, 0.11]

    fig1 = plt.figure()
    fig1.suptitle(title)
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

    if show_course:
        fig2 = plt.figure()
        fig2.suptitle('Course ' + str(course))
        axrho = fig2.add_subplot(111)
        axrho.plot(T, RHO)
        axrho.set_xlabel('t')
        axrho.set_ylabel('ρ')

    if plot_limit:
        ax1.set_ylim(xmax[0])
        ax2.set_ylim(xmax[1])
        ax3.set_ylim(xmax[2])
        ax4.set_ylim(xmax[3])
        ax5.set_ylim(xmax[4])
        axu.set_ylim(umax)

    plt.show()
