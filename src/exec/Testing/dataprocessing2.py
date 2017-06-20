#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:22:00 2017

@author: bradley

Data preprocessing module
"""
import GPy
import numpy as np
import sys
import os
import platform
# if(platform.system() != 'Darwin'):
#     os.environ["CUDA_VISIBLE_DEVICES"]="2"
#     import pycuda
from matplotlib import pyplot as plt
import math
import pandas as pd
import random
plt.style.use('ggplot')

def select_kernel(single_input):
     """ Selects a new kernel at random. Also checks if input is augmented or
     single. Single_input == 'False' by deafult. If changed will update the
     kernel .
     """
     kernel_list = ['StdPeriodic', 'RBF', 'MLP', 'Matern32' ]
     kernel_number = np.random.randint(0,4)

     # getattr allows us to get the related method from the class 
     kernel_name = kernel_list[kernel_number]
     
     # Randomly generate variance and length scale parameters
     var         = random.random()
     ls          = random.random()
     # module_name = 'pycuda'
     # if module_name in sys.modules:
     # 	use = True
     # else:
     # 	use = False

     if single_input and kernel_name != 'MLP':
         kernel = getattr(GPy.kern,kernel_list[kernel_number])(input_dim=1,\
                             variance=var, lengthscale = ls)
     elif single_input and kernel_name == 'MLP':
         kernel = getattr(GPy.kern,kernel_list[kernel_number])(input_dim=1,\
                             variance=var)
     else:
         if kernel_name != 'MLP':
             kernel = getattr(GPy.kern,kernel_list[kernel_number])(input_dim=2,\
                             variance=var,lengthscale = ls)
         else:
             kernel = getattr(GPy.kern,kernel_list[kernel_number])(input_dim=2,\
                             variance=var)

     return kernel, kernel_list[kernel_number],var,ls
def error(prediction, groundtruth):
    """Calculates the RMSE"""
    groundtruth = groundtruth[:prediction.shape[0]]
    assert np.size(prediction) == np.size(groundtruth)
    diff_square  = (prediction - groundtruth)**2
    error        =  np.mean(diff_square)
    nrmse        =  math.sqrt(error)/ (np.amax(groundtruth) - np.amin(groundtruth))
    return nrmse

def norm_data(data):
    """Takes training and test data and normalises it, to ensure zero mean
       and unit variance. It also ensures ||data|| = 1
       
       Input:  
       
       data       -  N x M np.array
           
       Output
       
       data_norm  - \mu(norm_data) = 0 \var(norm_data) = 1
    """
    mean_data = np.mean(data)
    std_data  = np.sqrt(np.var(data))
    data_norm = (data - mean_data)/(std_data)
    try:
        assert np.mean(data_norm)<0.00000001
        assert np.var(data_norm) > 0.9999999 and np.var(data_norm) < 1.0000001
    except:
        print(np.var(data_norm))
    return data_norm
def unit_data(data):
    norm      = sum(np.dot(data.T,data))
    scalar    = 1 / math.sqrt(norm)
    data_unit = data*scalar
    try:
        assert sum(data_unit**2) > 0.9999999 and sum(data_unit*2) <1.0000001
    except:
        print('Here lies the problem ' , sum(norm**2))
    return data_unit


def save_data(input_data,rmse,kernel_params,post_mean,post_var,parameters):
    no_points     = parameters[0]
    option        = parameters[1]
    model         = parameters[3]
    data_keys     = parameters[4]
    layer_no      = parameters[5]
    kernel_name   = parameters[6]

    var       = parameters[7][0]
    ls        = parameters[7][1]
    
    predicted = post_mean.flatten()
    actual    = input_data[data_keys[1]].flatten()
    time_pred = input_data['val'][0:no_points,0].flatten()
    time_all  = input_data['val'][:,0].flatten()
    cov_diag  = post_var.flatten()
    kernel_params = kernel_params.flatten()
    
    d  = dict(Time_pred = time_pred, Time = time_all,Predicted = predicted, Actual = actual, Diag_cov = cov_diag,Kernel_params = kernel_params, Kernels = kernel_name, NRMSE = rmse, lengthscale =ls, variance = var)
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
    PATH = '../Data/' +model+ '/data_layer_/' + option 
    file   = PATH + '/'+str(layer_no) +'.csv'
    os.makedirs(PATH,exist_ok = True)
    df.to_csv(file)
    
def plotting(data,rmse,parameters):
    """For plotting the graphs to verify the confidence in the prediction of
       our training data. Saves plots in ../../Plots. Takes the model, m
       and uses GPys inbuilt plotting. 
       Inputs:
           
           augX             - augX[:,0] posterior mean (post_mean)
                              augX[:,1] original inputs
           
            Y               - The given output for the training or test data
            
        post_cov            - The diagonal of cov(x*,x*) for plotting confidence
        

        
        Output:
            
         plots               - Saved as pdf to given file name 
   """
    no_points     = parameters[0]
    layer_no      = parameters[5]
    option        = parameters[1]
#    single_input  = parameters[2]
    model         = parameters[3]
    Xtrain = data[0]
    Ytrain = data[1]
    Xtest  = data[2]
    Ytest  = data[3]
    post_mean = data[4]
    post_var  = data[5]
    # temp1 = Xtest[no_points:]
    # Xplot = np.concatenate((Xtrain,temp1),axis=0)
    s  = np.sqrt(post_var.flatten())
    mu = post_mean.flatten()
    plt.figure(1)
    plt.clf()
    plt.hold(True)
    
    if model == 'Aug_Duvenaud' or model == 'Aug_inputs':
        plt.plot(Xtrain[:,1].flatten(),Ytrain.flatten(), 'r.',label = 'training data')
    elif model == 'Single_inputs':
        plt.plot(Xtrain.flatten(),Ytrain.flatten(), 'r.',label = 'training data')
    
    plt.plot(Xtest[no_points:,0].flatten(), Ytest[no_points:].flatten(), 'b.', label='test data')
    plt.fill_between(Xtest[:,0].flatten(), mu-2*s, mu+2*s, color="#C0C0C0", label = 'mu +/- 2sd')
    plt.plot(Xtest[:,0].flatten(), mu, 'w-', lw=1, label = 'Prediction')

    plt.legend(loc='upper right')
    plt.title("This is layer "+str(layer_no)+" with NRMSE  " + str(rmse))
    name_of_plot = 'Prediction_in_layer_' + str(layer_no)
    PATH = '../Plots/'+ model + '/'+ option
    os.makedirs(PATH,exist_ok = True)
    file = PATH + '/_' + name_of_plot +'_'+str(parameters[0]) +'.png'
    plt.savefig(file, format = 'png',dpi=600)


def plot_cov_ft(kernel,Xtrain,parameters):
    layer_no = parameters[5] 
    option   = parameters[1]
    model    = parameters[3]
    
    plt.figure(2)
    plt.clf()
    k = kernel.K(Xtrain)
    plt.imshow(k)
    plt.colorbar()
    name_of_plot = 'Plot_of_covariance_function_layer_ ' + str(layer_no)
    plt.title(name_of_plot)
    PATH = '../Plots/'+ model + '/'+ option
    os.makedirs(PATH,exist_ok = True)
    file = PATH + '/_' + name_of_plot+'_' +str(parameters[0]) +'.png'
    plt.savefig(file, format ='png',dpi=600)
