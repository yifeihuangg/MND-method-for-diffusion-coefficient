# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
import sys
import numbers
import time


def get_MB_potential(xycoord):
    """
    Evaluates the Muller-Brown potential

    Parameters
    ----------
    xycoord : arraylike
        An array-like structure with two elements, corresponding to the x and y coordinate of the point at which to evaluate the potential

    Returns
    -------
    V : float
        The value of the potential evaluated at xycoord[0],xycoord[1]
    """
    # Set Muller Brown Parameters
    A = np.array([-200.0,-100.0,-170.0,15])
    a = np.array([-1.,-1.,-6.5,0.7])
    b = np.array([0,0,11,0.6])
    c = np.array([-10.,-10,-6.5,0.7])
    x_0 = np.array([1.0,0.0,-0.5,-1])
    y_0 = np.array([0.0,0.5,1.5,1.0])
    
    # Evaluate the potential. We rename the 1st and 2nd coordinate to x and y.
    V = 0.0
    x = xycoord[0]
    y = xycoord[1]
    # The Muller Brown potential is calculated using a sum of 4 Gaussians.
    for i in xrange(4):
        V += A[i]*np.exp(a[i]*(x-x_0[i])**2+b[i]*(x-x_0[i])*(y-y_0[i])+c[i]*(y-y_0[i])**2)
    # Reduce size to make kT=1 reasonable, set minimum to (approximately) zero
    V = V/20. + 7.352119 
    return V

def get_MB_force(xycoord):
    """
    Evaluates the Muller-Brown potential

    Parameters
    ----------
    xycoord : arraylike
        An array-like structure with two elements, corresponding to the x and y coordinate of the point at which to evaluate the potential

    Returns
    -------
    f : array
        A numpy array with two elements.  The first is the force in the x direction, and the second is the force in the y.
    """
    # Set Muller Brown Parameters
    A = np.array([-200.0,-100.0,-170.0,15])
    a = np.array([-1.,-1.,-6.5,0.7])
    b = np.array([0,0,11,0.6])
    c = np.array([-10.,-10,-6.5,0.7])
    x_0 = np.array([1.0,0.0,-0.5,-1])
    y_0 = np.array([0.0,0.5,1.5,1.0])
    
    # Evaluate the potential. We rename the 1st and 2nd coordinate to x and y.
    fx=0.0
    fy=0.0
    x=xycoord[0]
    y=xycoord[1]
    # The Muller Brown potential is defined as a sum of 4 Gaussians.
    # To calculate the force, we add up the negative partial deratives 
    # in both x and y.
    # This becomes the x and y components of our force, respectively.
    for i in xrange(4):
        factor=-1.0*A[i]* np.exp(a[i]*(x-x_0[i])**2+b[i]*(x-x_0[i])*(y-y_0[i])+c[i]*(y-y_0[i])**2)
        fx+=(2.0*a[i]*(x-x_0[i])+b[i]*(y-y_0[i]))*factor
        fy+=(2.0*c[i]*(y-y_0[i])+b[i]*(x-x_0[i]))*factor
    return np.array([fx,fy])/20.

def brownian_dynamics(nsteps,x0,force_method,dt=0.001,D=1.,kT=1.0):
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
    D : float or 2D array-like, optional
        Diffusion tensor for the system.  If a float is given, the diffusion tensor is taken to be a diagonal matrix with that float on the diagonal.  Default is 1, or the identity matrix.
    kT : float, optional
        Boltzmann factor for the system (k_B * T).  Default is natural units (1.0)

    Returns
    -------
    traj : 2D array
        Two dimensonal array, where the element i,j is the j'th coordinate of the system at timestep i.

    """
    # Set defaults and parameters
    ndim = len(x0) # Find dimensionality of the system
    if isinstance(D,numbers.Number):
        D = np.eye(ndim)*D # If scalar provided, convert to array.
    else:
        D = np.array(D) # Typecast to array for easy math.
    c = spl.cholesky(D) # Square root of Diffusion matrix hits the noise
    sig = np.sqrt(2.*dt) # Perform some algebra ahead of time.

    # Propagate Brownian dynamics according to the Euler-Maruyama method.
    traj = []    
    cfg = np.copy(x0) 
    for j in xrange(int(nsteps)):
        rando = np.dot(c,np.random.randn(ndim))
        force = np.dot(D,force_method(cfg))
        cfg += dt * force + sig * rando
        traj.append(np.copy(cfg))
    return np.array(traj) 

def main():
    # Sample Muller Brown potential
    x0 = np.array([-.5,1.5])
    D = np.array([[2.,-1.],[-1.,2.]])
    traj = brownian_dynamics(100000,x0,get_MB_force)
    # Keep every tenth point
    traj = traj[::10]
    # Plot trajectory
    plt.scatter(traj[:,0],traj[:,1])
    plt.show()
    # Plot Free Energy (-log(density))
    hist,edges_x,edges_y = np.histogram2d(traj[:,0],traj[:,1],range=((-2,2),(-2,2)),bins=40,normed=True)
    fe = -np.log(hist)
    fe -= np.min(fe)
    YY,XX = np.meshgrid(edges_x,edges_y)
    plt.pcolor(XX,YY,fe,cmap='viridis',vmin=0,vmax=6)
    plt.colorbar()
    plt.show()
    

    
if __name__ == "__main__":
    main()
