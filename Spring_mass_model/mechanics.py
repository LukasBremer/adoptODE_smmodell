import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree

import numpy as np
import scipy
import matplotlib.pyplot as plt

# grid points xj(0 to 3): Mass point x
#      xj0 
#
# xj3  x  xj1
#
#      xj2
#
# shape(x)  = (N,2,1)
# shape(xj) = (N,2,4)

# active axial points q 0 to 3; Mass point x; eta > 1/2
#
#              0
#              q0\    
#        xcm0\    \xcm1
#         /   \q1   /  
#  3  q0p/ /q1p x q2p  /q3p 1 
#         /    q2\    /  
#        xcm3\    \xcm2     
#             \q3
#              2
#
# shape(q)  = (N,2,4)


@jit
def total_force(x, x_j, x_cm, l_a, t, params):
    """defines the total force of the system"""
    
    return grid_force(x, x_j, params) + axial_force_a(x, x_j, x_cm, l_a, params) + axial_force_p(x, x_j, x_cm, params)

@jit
def grid_force(x,x_j,params):
    """defines the force that is given by the grid of the system"""
    local_force = 0
    k_g = params['k_g']
    l_g = params['l_g']

    for xj in x_j:
        local_force += f_ij(x, xj, k_g, l_g)
    return local_force

@jit
def axial_force_a(x, x_j, x_cm, l_a, params):
    """defines the force that is given by the active axial springs of the system"""
    local_force = 0
    eta = params['eta']
    k_a = params['k_a']
    l_g = params['l_g']

    q_j = interpolate_q(x_j, params['eta'],"a")
    
    local_force += f_ij(q_j[:,0], x_cm[:,1], k_a, l_a)*(eta)
    local_force += f_ij(q_j[:,1], x_cm[:,2], k_a, l_a)*(1-eta)
    local_force += f_ij(q_j[:,2], x_cm[:,3], k_a, l_a)*(1-eta)
    local_force += f_ij(q_j[:,3], x_cm[:,0], k_a, l_a)*(eta)

    return local_force

@jit 
def axial_force_p(x, x_j, x_cm, params):
    """defines the force that is given by the passive axial springs of the system"""
    local_force = 0
    eta = params['eta']
    k_p = params['k_p']
    l_p = params['l_p']
    l_g = params['l_g']
    
    q_j = interpolate_q(x_j, params['eta'],"p")

    local_force += f_ij(q_j[:,0], x_cm[:,0], k_p, l_p)*(eta)
    local_force += f_ij(q_j[:,1], x_cm[:,3], k_p, l_p)*(1-eta)
    local_force += f_ij(q_j[:,2], x_cm[:,1], k_p, l_p)*(1-eta)
    local_force += f_ij(q_j[:,3], x_cm[:,2], k_p, l_p)*(eta)
    
    return local_force

@jit 
def interpolate_q(x_j, x, eta, l_g, q_type):
    """interpolates the axial points q"""
    q = jnp.zeros((2,4))
    if q_type == "a":
        q[0] = x + x_j[:,0]-x * eta
        q[1] = x + x_j[:,0]-x * (1-eta)
        q[2] = x + x_j[:,2]-x * (1-eta)
        q[3] = x + x_j[:,2]-x * eta    
    elif q_type == "p":
        q[0] = x + x_j[:,3]-x * eta 
        q[1] = x + x_j[:,3]-x * (1-eta)
        q[2] = x + x_j[:,1]-x * (1-eta)
        q[3] = x + x_j[:,1]-x * eta
        
    return q
        

@jit
def f_ij(x_i, x_j, k, l_0):
    """returns the force from j to i"""
    return k*(jnp.linalg.norm(x_j-x_i)-l_0)*e_i_to_j(x_i, x_j)

@jit        
def e_i_to_j(x_i, x_j):
    """"returns normal vector from x_i to x_j"""
    return (x_j-x_i)/jnp.linalg.norm(x_j-x_i)

@jit
def interpolate_spline(x, t):

    i = jnp.abs(t_eval - t).argmin()
    # Check if i is within the valid range
    i = round()
    if i < 1:
        i=+1
    elif i >= len(x) - 1:
        i -= 1
        
    # Extract the neighboring points
    x_prev = x[i-1]
    x_curr = x[i]
    x_next = x[i+1]

    # Create the spline interpolation object
    t_spline = jnp.array([t_eval[i-1],t_eval[i],t_eval[i+1]])
    spline = CubicSpline(t_spline, jnp.array([x_prev, x_curr, x_next]))

    return spline(t)