import jax.numpy as jnp
from jax import jit
from jax.flatten_util import ravel_pytree

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.interpolate
import interpax

# grid points xj(0 to 3): Mass point x
#      xj0 
#
# xj3  x   xj1
#
#      xj2
#
# shape(x)  = (N,2,1)
# shape(xj) = (N,2,4)

# active/passive axial points q 0 to 3; Mass point x; eta > 1/2
#
#              0
#             q0a\    
#        xcm0\    \xcm1
#         /   \q1a  /  
#  3   q0/ /q1  x q2   /q3  1 
#         /   q2a\    /  
#        xcm3\    \xcm2     
#             \q3a
#              2
#
# shape(q)  = (N,2,4)

#pure mechanics
@jit
def total_force(x, x_j, x_cm, l_a, t, params):
    """defines the total force of the system"""
    f = grid_force(x, x_j, params) + axial_force_a(x, x_j, x_cm, l_a, params) + axial_force_p(x, x_j, x_cm, params)
    return f

@jit
def grid_force(x,x_j,params):
    """defines the force that is given by the grid of the system"""
    local_force = jnp.zeros(2)
    k_g = params['k_g']
    l_g = params['l_g']

    for i in range(4):
        #local_force = local_force.at[:].set(local_force + f_ij(x, x_j[i,:], k_g, l_g)) 
        local_force += f_ij(x, x_j[i,:], k_g, l_g)
        
    return local_force

@jit
def axial_force_p(x, x_j, x_cm, params):
    """defines the force that is given by the active axial springs of the system"""
    local_force = jnp.zeros(2)
    eta = params['eta']
    k_p = params['k_p']
    l_p = params['l_p']

    q_j = interpolate_q_p(x_j, x, eta)

    local_force += f_ij(q_j[0,:], x_cm[0,:], k_p, l_p)*(1-eta)
    local_force += f_ij(q_j[1,:], x_cm[3,:], k_p, l_p)*(eta)
    local_force += f_ij(q_j[2,:], x_cm[1,:], k_p, l_p)*(eta)
    local_force += f_ij(q_j[3,:], x_cm[2,:], k_p, l_p)*(1-eta)

    return local_force
    
@jit 
def axial_force_a(x, x_j, x_cm,l_a, params):
    """defines the force that is given by the passive axial springs of the system"""
    local_force = jnp.zeros(2)
    eta = params['eta']
    k_a = params['k_a']
    
    q_j = interpolate_q_a(x_j, x, eta)

    local_force += f_ij(q_j[0,:], x_cm[1,:], k_a, l_a[1])*(1-eta)
    local_force += f_ij(q_j[1,:], x_cm[0,:], k_a, l_a[0])*(eta)
    local_force += f_ij(q_j[2,:], x_cm[2,:], k_a, l_a[2])*(eta)
    local_force += f_ij(q_j[3,:], x_cm[3,:], k_a, l_a[3])*(1-eta)
    
    return local_force

@jit
def interpolate_q_a(x_j, x, eta):
    """interpolates the axial points q"""
    q = jnp.zeros((4,2))
    
    q = q.at[0,:].set(x + (x_j[0,:]-x) * eta)
    q = q.at[1,:].set(x + (x_j[0,:]-x) * (1-eta))
    q = q.at[2,:].set(x + (x_j[2,:]-x) * (1-eta))
    q = q.at[3,:].set(x + (x_j[2,:]-x) * eta) 

    return q

@jit
def interpolate_q_p(x_j, x, eta):  
    """interpolates the axial points q"""
    q = jnp.zeros((4,2))

    q = q.at[0,:].set(x + (x_j[3,:]-x) * eta)
    q = q.at[1,:].set(x + (x_j[3,:]-x) * (1-eta))
    q = q.at[2,:].set(x + (x_j[1,:]-x) * (1-eta))
    q = q.at[3,:].set(x + (x_j[1,:]-x) * eta)
    
        
    return q

@jit
def f_ij(x_i, x_j, k, l_0):
    """returns the force from acting on i"""
    return k*(jnp.linalg.norm(x_j-x_i)-l_0)*e_i_to_j(x_i, x_j)

@jit        
def e_i_to_j(x_i, x_j):
    """"returns normal vector from x_i to x_j"""
    return (x_j-x_i)/jnp.linalg.norm(x_j-x_i)


