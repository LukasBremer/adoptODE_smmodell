import numpy as np
import csv
import configparser
import matplotlib.pyplot as plt
import progressbar
import scipy
from jax import jit
import jax.numpy as jnp

def read_config(variables, mode='chaos',file='config.ini'):
    config = configparser.ConfigParser()
    config.read(file)

    if variables != []:
        for i in range(len(variables)):
            variables[i] = float(config[mode][variables[i]])

    global N,size    
    size = int(config[mode]['size'])
    N_max = float(config[mode]['N_max'])
    N_output = float(config[mode]['N_output'])
    sample_rate = float(config[mode]['sample_rate'])
    N = int((N_max - N_output)/sample_rate)
    return N,size,variables
    
def read_scalar(file,shape_of_data):
    #reads the data that is produced by PrintArray from file arrayhandling.c
    data = np.empty(shape_of_data)
    mech_reader = csv.reader(open(file, "r"), delimiter=",")
    print("shape of data: ",data.shape)

    bar = progressbar.ProgressBar(maxval=N, left_justify=True)
    bar.start()
    i = 0
    k = 0
    tot = 0
    for mech_line in mech_reader:
        if not mech_line:
            i += 1
            k = 0
            bar.update(i)
        else:
            arr_mech = np.array(list(mech_line))
            data[0,i,k] = arr_mech
            k += 1
            tot += 1
    bar.finish()
    return data

def read_vector(file,shape_of_data):

    mech_reader = csv.reader(open(file, "r"), delimiter=",")
    i,j,k,t_n=0,0,0,0
    data = np.empty(shape_of_data)
    print("shape of data: ",data.shape)

    bar = progressbar.ProgressBar(maxval=N, left_justify=True)
    bar.start()

    for mech_line in mech_reader:
        #print(len(mech_line))
        if len(mech_line) == 0:
            i += 1
            k = 0
        elif len(mech_line) == 1:
            k = 0
            i = 0
            t_n += 1
            bar.update(t_n)
        else:
            arr_mech = np.array(list(mech_line),float)
            data[t_n,k,i,:] = arr_mech               
            #print(t_n,k,i)
            k += 1

    bar.finish()
    return data

def shape_input_for_adoptode(x, x_cm, T_a, i, j):


    # extracts the i,j coordinate from x and the 4 nearest neighbors from x_cm
    N,size,ls = read_config(["l_0","c_a",'pad'])
    l_a0, c_a, pad = ls
    T_a_arr = jnp.array([T_a[:,i-1,j-1],T_a[:,i-1,j],T_a[:,i,j],T_a[:,i,j-1]])
    print(T_a_arr.shape)
    l_a_i = l_a0/(1+ c_a*T_a_arr)
    x_i = x[:,:,i,j]
    x_j = np.array([x[:,:,i+1,j], x[:,:,i,j+1], x[:,:,i-1,j], x[:,:,i,j-1] ])
    x_cm_i = np.array([x_cm[:,:,i,j-1], x_cm[:,:,i,j], x_cm[:,:,i-1,j], x_cm[:,:,i-1,j-1] ])
    return x_i,x_j, x_cm_i,l_a_i

@jit
def t_to_value_4p(x,t_int,t):
    
    delta_t = (t_int[-1]-t_int[0])/(len(t_int))
        
    i = jnp.rint(t/delta_t).astype(int)
    
    return x[:,i,:]

@jit
def t_to_value_1p(x,t_int,t):
    delta_t = (t_int[-1]-t_int[0])/(len(t_int))
        
    i = jnp.rint(t/delta_t).astype(int)
    
    return x[:,i]


def interpolate_x(x,t_eval,m):
    
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
    
    x_int = np.zeros((4,len(t_eval)*m))
    interp_points = len(t_eval)*m
    # Generate interpolated values
    for i in range(4):    
        cs0 = scipy.interpolate.CubicSpline(t_eval,arr[i,:])
        t_interp = np.linspace(t_eval[0], t_eval[-1], interp_points)  
        x_int[i,:] = cs0(t_interp)


    return t_interp, x_int
