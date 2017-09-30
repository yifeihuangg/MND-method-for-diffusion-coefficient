#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import kernel_regression as kr
import math
import matplotlib.patches as mpatches
import sys

"""
Generate trajctories based on Brownian dynamics
"""

def get_U(x):
    """
    Returns the potential energy of a function that is a quartic double well of form (x^4-2x^2).

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.
    a : float
        The prefactor for the potential.

    Returns
    -------
    U : float
        Value of the potential.
    """
    xsq = x*x
    print xsq, xsq*xsq, 2*xsq
    return xsq*xsq-2.*xsq

def get_F(x):
    """
    Returns the potential energy of a function that is a quartic double well of form (x^4-2x^2).

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    F : float
        Value of the force.
    """
    return -(4*x*x*x-4*x)

def get_D(x):
    """
    Returns the value of the diffusion function at x.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    D : float
        Value of the diffusion function.
    """
    return np.sin(x)*0.5+1
#    return np.sin(-x)*0.5+1

def get_dD(x):
    """
    Returns the value of the divergence of the diffusion function at x.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    dD : float
        Value of the divergence of the diffusion function.
    """
    return np.cos(x)*0.5
#    return np.cos(-x)*0.5

def brownian_dynamics(nsteps,x0,force_method,get_divD,get_D,dt=0.001,kT=1.0):
    """
    Runs brownian dynamics.

    Parameters
    ----------
    nsteps : int
        Number of dynamics steps to run.
    x0 : 1d array-like
        Starting coordinate from which to run the dynamics.
    force_method : subroutine
        Subroutine that yields the force of the system.  Must take in an array of the same shape as x0, and return a force vector of the same size.
    dt : float, optional
        Timestep for the dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.
    get_D : Subroutine
         Subroutine that yields diffusion tensor for the system.
    get_divD: Subroutine
         Subroutine that yields the divergence of D
    kT : float, optional
        Boltzmann factor for the system (k_B * T).  Default is natural units (1.0)


    Returns
    -------
    traj : 2D array
        Two dimensonal array, where the element i,j is the j'th coordinate of the system at timestep i.

    """
    # Set defaults and parameters
    ndim = len(x0) # Find dimensionality of the system
    # Propagate Brownian dynamics according to the Euler-Maruyama method.
    traj = []
    cfg = np.copy(x0)
    sig = np.sqrt(2.* dt) # Perform some algebra ahead of time.
    for j in xrange(int(nsteps)):
        D = get_D(cfg) # Typecast to array for easy math.
        c = np.sqrt(D) # Square root of Diffusion matrix hits the noise
        rando = np.dot(c,np.random.randn(ndim))
        force = np.dot(D,force_method(cfg))
        divD = get_divD(cfg)
        cfg += dt * force + sig * rando + divD * dt
        traj.append(np.copy(cfg))
    return np.array(traj)





"""
Below is two estimations
"""

def get_b_KDE(trajs):
    """
    Estimates the E(dx) using KDE method.
    Parameters
    ----------
    trajs: list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.

    Returns
    -------
    model : function(?)
        Estimated model of E(dx).
    """
    z_trajs = np.zeros([len(trajs),len(trajs[0])-1])
    x_trajs = np.zeros_like(z_trajs)
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    # Store all the y in one dataset and use all the x's and corresponding y's to make model
    for n in xrange(len(trajs)):
        for i in xrange(len(trajs[0])-1):
            z_trajs[n][i] = (trajs[n][i+1] - trajs[n][i])
            x_trajs[n][i] = trajs[n][i]
    # Put all training datas in 1d array
    x = x_trajs.flatten()
    z = z_trajs.flatten()
    # Kernel.fit only takes in input matrix with shape(n,1), so change it into (n,1)
    x.resize((len(x),1))
    model = kernel.fit(x,z)
    return model

def get_D_KDE(trajs, pos, subsampling=10, dt=0.001,use_drift=True):
    """
    Estimates the position-dependent diffusion constant using KDE method.
    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    pos: list
        Positions where you would like to estimate diffusion coefficient using the model
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 10.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.
    use_drift: optional, True/False
         If True, use drift term in the approximation, if False, not use.
    Returns
    -------
    D : array
        Estimate of the diffusion constant corresponding to pos.
    """
    y_trajs = np.zeros([len(trajs),len(trajs[0])-1])
    x_trajs = np.zeros_like(y_trajs)
    D = np.zeros(len(pos))
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    if use_drift:
        model = get_b_KDE(trajs)
    # Store all the y in one dataset and use all the x's and corresponding y's to make model
    for n in xrange(len(trajs)):
        if use_drift:
            b_dt = model.predict(trajs[n])
        for i in xrange(len(trajs[0])-1):
            y_trajs[n][i] = ((trajs[n][i+1] - trajs[n][i])**2)
            if use_drift:
                y_trajs[n][i] -= (b_dt[i])**2
            x_trajs[n][i] = trajs[n][i]
    # Put all training datas in 1d array
    x = x_trajs.flatten()
    y = y_trajs.flatten()
    # Kernel.fit only takes in input matrix with shape(n,1), so change it into (n,1)
    x.resize((len(x),1))
    model = kernel.fit(x,y)
    # model.predict only take (x,1) size input
    pos.resize((len(pos),1))
    D = (model.predict(pos))/(dt*2*subsampling)
    return D
def get_D_hist_MND(trajs,hist_edges,subsampling,t,dt = 0.001):
    """Estimates the position-dependent diffusion constant using the correlation function of Ma, Nag, Dinner at time t.

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    hist_edges : array
        Edges of each histogram bin for assigning the diffusion constantes.
    subsampling: float
        The subsampling used when calculate trajectories.
    t : int
        Time lag over which to calculate the correlation function.
    dt : float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.

    Returns
    -------
    D_t : array
        Estimate of the diffusion constant at time t for each histogram bin.
    """
    # Initialize array of Diffusion Constants
    D_t = np.zeros(len(hist_edges)-1)
    binwidth = round(hist_edges[1]-hist_edges[0],4)
    bin_num = len(hist_edges)-1
    hist_left_edge = hist_edges[0]
    # Calculate the average correlation function for each histogram bin.
    # I think the easiest way to do this is to have a list for each histogram
    # bin (you might need a list of lists), and append the values to the correct
    # lists.  Once you are done, you then calculate the average of the list.
    keyDict = range(bin_num)
    C_anit = dict([(key,[]) for key in keyDict])# use dictionary to store each list of C (with same location a)
    for traj in trajs:
        L = len(traj)
        for i in xrange(L-t):
            ## Evaluate X dot, and value of correlation, and assign to appropriate bin.
            a = (traj[i+t]+traj[i])/2
            x = math.floor((a-hist_left_edge)/binwidth)
            if ((hist_edges[-1]-hist_edges[0])/binwidth) > x >= 0:
                C_anit[x].append(((traj[i+1]-traj[i-1])/(2*subsampling*dt))*(traj[i+t]-traj[i])) # value of correlation,for x dot I use central difference to calculate velocity
            else:
                continue

    ## Calculate the average in each histogram bin.
    for x,c in C_anit.iteritems():
        if len(c)!= 0:
            D_t[x] = sum(c)/float(len(c))
        else:
            D_t[x] = 0

    return D_t


def get_D_MND(trajs,hist_edges,subsampling, Tmax,Tmin=1,dt = 0.001):
    """Calculates the position-dependent diffusion constant in 1D, using the method given by Ma, Nag, and Dinner, JCP 2006.

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    hist_edges : array
        Edges of each histogram bin for assigning the diffusion constantes.
    Tmax : int
        Maximum time lag over which to calculate the diffusion constant, in units of trajectory timesteps.
    Tmin : int, optional
        Minimum time lag over which to calculate the diffusion constant, in units of trajectory timesteps.  Default is 1.

    Returns
    -------
    D : array
        Estimate of the diffusion constant for each histogram bin.
    """
    # Check that trajectories are two dimensional.
    for i,traj in enumerate(trajs):
        if len(np.shape(traj)) != 2:
            raise ValueError('Trajectory %d was not two-dimensional.  Note that if you have one-dimensional trajectory trj, you can make it two-dimensional using np.transpose([trj]). '%(i+1))

    # Calculate Diffusion constant for each histogram bin.
    D_hists = []
    for t in xrange(Tmin,Tmax):
        D_t = get_D_hist_MND(trajs,hist_edges,subsampling,t,dt=dt)
        D_hists.append(D_t)
    D_hists = np.array(D_hists)

    # Find the maximum over each time
    ## NEED CODE HERE
    D = np.amax(D_hists, axis = 0)# find max along col when axis =0, and return the max for each col
    ## Fix this to return the diffusion constant
    return D





"""
Below is main
"""

def main():
    # Load in trajectories into trajs file.
    trajs = []
    initial = np.linspace(-2.,2.,101)
    kT = 1.0
#    dt = 0.00001
##    nsteps = 500
#    nsteps = 10000
#    subsampling = 1000
    dt = 0.001
    nsteps = 100
#    nsteps = 100
    subsampling = 10
    for inipoint in initial:
        for times in xrange(5):
            x0 =  np.array([inipoint])
            traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
            # time consuming -> from begining to end skip every 10 points
            traj = traj[::subsampling]
            trajs.append(traj)
    # Define maximum and minimum lag time below
    Tmax = 10
    Tmin = 1
    # Define histogram edges
    hist_edges = np.linspace(-2.,2.,51)
    # positions used to estimate true D, use middle point of each histogram bin
    pos = np.zeros(len(hist_edges)-1)
    for m in xrange(len(hist_edges)-1):
        pos[m] = (hist_edges[m]+hist_edges[m+1])/2

    D_wo_drift_KDE= get_D_KDE(trajs,pos,subsampling,dt,use_drift=False)
    D_w_drift_KDE= get_D_KDE(trajs,pos,subsampling,dt)
    D_MND = get_D_MND(trajs,hist_edges,subsampling,Tmax,Tmin, dt=dt)
    D_true = get_D(pos)

    # Save all the estimations and true D values
    rank = int(sys.argv[1])
    np.savetxt('D_wo_drift_KDE_%d.txt' %rank, D_wo_drift_KDE)
    np.savetxt('D_w_drift_KDE_%d.txt' %rank, D_w_drift_KDE)
    np.savetxt('D_MND_%d.txt' %rank, D_MND)
    print("DONE!")
    #Plot estimation and true D values
    plt.figure()
#   plt.ylim(-1.5, 5)
    plt.plot(pos, D_w_drift_KDE,c = 'y', label = 'kernel w drift')
    plt.plot(pos, D_wo_drift_KDE,c = 'r', label = 'kernel w/o drift')
    plt.plot(pos, D_MND, c = 'g', label = 'MND')
    plt.scatter(pos, D_true, label = 'true')
    plt.legend(loc = 2)
    plt.xlabel('Positions')
    plt.ylabel('Diffusion coefficient(D)')
    plt.title('Comparison of estimations using different methods')
    plt.savefig('Compare%d'%rank)
    plt.show()


if __name__ == "__main__":
    main()
