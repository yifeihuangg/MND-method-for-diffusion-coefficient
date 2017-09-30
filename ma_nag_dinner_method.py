# -*- coding: utf-8 -*-
import numpy as np
import math 
import matplotlib.pyplot as plt

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



def get_D_hist(trajs,hist_edges,t,dt = 0.001):
    """Estimates the position-dependent diffusion constant using the correlation function of Ma, Nag, Dinner at time t. 

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    hist_edges : array
        Edges of each histogram bin for assigning the diffusion constantes.
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
            if 100 > x >= 0:
                C_anit[x].append(((traj[i+1]-traj[i-1])/(2*dt))*(traj[i+t]-traj[i])) # value of correlation,for x dot I use central difference to calculate velocity    
            else:
                continue
                
    ## Calculate the average in each histogram bin.
    for x,c in C_anit.iteritems(): 
        if len(c)!= 0: 
            D_t[x] = sum(c)/float(len(c))
        else:
            D_t[x] = 0
        
    return D_t


def diff_constant(trajs,hist_edges,Tmax,Tmin=1,dt = 0.001):
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
        D_t = get_D_hist(trajs,hist_edges,t,dt=dt)
        D_hists.append(D_t)
    D_hists = np.array(D_hists)

    # Find the maximum over each time
    ## NEED CODE HERE
    D = np.amax(D_hists, axis = 0)# find max along col when axis =0, and return the max for each col
    ## Fix this to return the diffusion constant
    return D

def main():
    # Load in trajectories into trajs file.
    trajs = []
    ## NEED CODE HERE
    initial = np.linspace(-2.,2.,101)
    for inipoint in initial:
        for times in xrange(50):
#            skip = 10000
#            nsteps = 50000 + skip
            nsteps = 1000 
            subsampling = 10
            x0 =  np.array([inipoint])
            kT = 1.0 
            dt = 0.001
            traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT) 
            # burn-in trajectory drop the first skip = 10000 data points
            #traj = traj[-(nsteps-skip):]
            # time consuming -> from begining to end skip every 10 points 
            traj = traj[::subsampling]
            trajs.append(traj)
  
       
   
    # Define maximum lag time below 
    Tmax = 10
    
    # Define the edges of each histogram bin.
    # For the example below, bin 1 goes from -2. to -1.96
    hist_edges = np.linspace(-2.,2.,101)

    D = diff_constant(trajs,hist_edges,Tmax,dt=dt*10)
    
    
    pos = np.zeros(len(hist_edges)-1)
    for m in xrange(len(hist_edges)-1):
        pos[m] = (hist_edges[m]+hist_edges[m+1])/2
        
    #plot the true D and estimation D
    fig = plt.figure()    
    plt.plot(pos,D,c = 'y', label = 'MND method')
    plt.hold('on')
    plt.plot(pos, get_D(pos), label = 'true')
    plt.xlabel('position')
    plt.ylabel('Diffusion coefficient(D)')
    plt.title('Ma Nag Dinner Method')
    fig.savefig('MNDplot')
    plt.legend()
    
    
    print('Done!')
if __name__ == "__main__":
    main()
