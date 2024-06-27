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

# active axial points q 0 to 3; Mass point x; eta > 1/2
#
#              0
#              q0\    
#        xcm0\    \xcm1
#         /   \q1   /  
#  3  q0a/ /q1a x q2a  /q3a 1 
#         /    q2\    /  
#        xcm3\    \xcm2     
#             \q3
#              2
#
# shape(q)  = (N,2,4)

#pure mechanics
@jit
def total_force(x, x_j, x_cm, l_a, t, params):
    """defines the total force of the system"""
    
    return grid_force(x, x_j, params) + axial_force_a(x, x_j, x_cm, l_a, params) + axial_force_p(x, x_j, x_cm, params)

@jit
def grid_force(x,x_j,params):
    """defines the force that is given by the grid of the system"""
    local_force = jnp.zeros(2)
    k_g = params['k_g']
    l_g = params['l_g']

    for xj in x_j:
        local_force = local_force.at[:].set(local_force + f_ij(x, xj, k_g, l_g)) 
    
    return local_force

@jit
def axial_force_p(x, x_j, x_cm, params):
    """defines the force that is given by the active axial springs of the system"""
    local_force = jnp.zeros(2)
    eta = params['eta']
    k_a = params['k_p']
    l_p = params['l_p']

    q_j = interpolate_q_p(x_j, x, eta)

    local_force += f_ij(q_j[0,:], x_cm[1,:], k_a, l_p)*(eta)
    local_force += f_ij(q_j[1,:], x_cm[2,:], k_a, l_p)*(1-eta)
    local_force += f_ij(q_j[2,:], x_cm[3,:], k_a, l_p)*(1-eta)
    local_force += f_ij(q_j[3,:], x_cm[0,:], k_a, l_p)*(eta)

    return local_force
    
@jit 
def axial_force_a(x, x_j, x_cm,l_a, params):
    """defines the force that is given by the passive axial springs of the system"""
    local_force = jnp.zeros(2)
    eta = params['eta']
    k_a = params['k_a']
    
    q_j = interpolate_q_a(x_j, x, eta)

    local_force += f_ij(q_j[0,:], x_cm[0,:], k_a, l_a)*(eta)
    local_force += f_ij(q_j[1,:], x_cm[3,:], k_a, l_a)*(1-eta)
    local_force += f_ij(q_j[2,:], x_cm[1,:], k_a, l_a)*(1-eta)
    local_force += f_ij(q_j[3,:], x_cm[2,:], k_a, l_a)*(eta)
    
    return local_force

@jit
def interpolate_q_p(x_j, x, eta):
    """interpolates the axial points q"""
    q = jnp.zeros((4,2))
    
    q = q.at[0,:].set(x + x_j[0,:]-x * eta)
    q = q.at[1,:].set(x + x_j[0,:]-x * (1-eta))
    q = q.at[2,:].set(x + x_j[2,:]-x * (1-eta))
    q = q.at[3,:].set(x + x_j[2,:]-x * eta) 

    return q

@jit
def interpolate_q_a(x_j, x, eta):  
    """interpolates the axial points q"""
    q = jnp.zeros((4,2))

    q = q.at[0,:].set(x + x_j[3,:]-x * eta)
    q = q.at[1,:].set(x + x_j[3,:]-x * (1-eta))
    q = q.at[2,:].set(x + x_j[1,:]-x * (1-eta))
    q = q.at[3,:].set(x + x_j[1,:]-x * eta)
    
        
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
def t_to_value_4p(x,t_eval,t,N_int):
    # u = x-t
    # t_close = jnp.nonzero(np.where(u == jnp.min(jnp.absolute(u)),x,0))
    # #t_close = jnp.abs(u).min() + t
    delta_t = (t_eval[-1]-t_eval[0])/(len(t_eval)*N_int)
        
    i = jnp.rint(t/delta_t).astype(int)
    return x[:,i,:]

@jit
def t_to_value_1p(x,t_eval,t,N_int):
    # u = x-t
    # t_close = jnp.nonzero(np.where(u == jnp.min(jnp.absolute(u)),x,0))
    # #t_close = jnp.abs(u).min() + t
    delta_t = (t_eval[-1]-t_eval[0])/(len(t_eval)*N_int)
        
    i = jnp.rint(t/delta_t).astype(int)
    return x[i]

def interpolate_x(x,t_eval,m):
    if x.shape[0] == 4:
        x_int = np.zeros((4,len(t_eval)*m,2))
        for i in range(4):
            t_int, x_int[i] = interpolate_spline(x[i],t_eval,m)
        
    return t_int, x_int

def interpolate_spline(arr, t_eval,m):
    # Assuming x and y are the input data points
    y0_eval = arr[:,0]
    interp_points = len(t_eval)*m
    cs0 = scipy.interpolate.CubicSpline(t_eval,y0_eval)
    # Generate interpolated values
    t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
    y0_interp = cs0(t_interp)

    # Assuming x and y are the input data points
    y1_eval = arr[:,1]
    interp_points = len(t_eval)*m   
    cs1 = scipy.interpolate.CubicSpline(t_eval,y1_eval)
    # Generate interpolated values
    t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
    y1_interp = cs1(t_interp)

    return t_interp, jnp.array([y0_interp,y1_interp]).T # transpose to get the shape as input


def interpolate_scalar(arr, t_eval,m):
    # Assuming x and y are the input data points
    y0_eval = arr
    interp_points = len(t_eval)*m
    cs0 = scipy.interpolate.CubicSpline(t_eval,y0_eval)
    # Generate interpolated values
    t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
    y0_interp = cs0(t_interp)


    return t_interp, y0_interp
