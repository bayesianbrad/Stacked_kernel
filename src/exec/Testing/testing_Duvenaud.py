#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:59:44 2017

@author: bradley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:26:36 2017

@author: bradley
1input dimension:  Deep GP for testing
Pseudo code\diagram:
    
input X, k1(x,x), h (hidden targets), x* test input

---------------- X -------------------
                 |
                 K1 | X hyperparams \theta1
                 |
                 K1, K2 | X kernel = K1 + K2 hyperparams \theta1 and \theta2
                 |
                 K1, K2, K3 | X kernel = K1 + K2 + K3 hyperparams \theta1, \theta2, \theta3
                 |
                 :
                 :    
When to stop: Using auto-correlation when the difference between 
the posterior mean anbd the orginba; data , is around the 0 axis, i.e
the abs(input data - posterior mean of layer) < 1

    } 

    For testing: 
        
    Within fucntions * indicates maybe no necessary
"""


import GPy
from matplotlib import pyplot as plt
import numpy as np
import math
import pylab
import random
# np.random.seed (seed = 0)
plt.style.use('ggplot')
try:
    matplotlib.pyplot.close("all")
except:
    pass

def regression(Xsamp,Y,Xpred,kernel, layer_no, sigma_noise):
    """GP regression module - uses GPy
       - Generates a new kernel, made from the previous
       kernel. 
       - Carries out GP regression + optimisation
       - Does the prediction for the output of the layer
       
       Inputs:
           
       Xsamp     - R x 2 - Sampled from original augmented input augX,
                           R is the number of samples
       Y         - R x 1 - Generated from a MVN N(0, f(augX))
                           (actually f(Xsamp))
       X_pred    - N x 2 - For finacial data, next set of times.To add later
                   N x 2 - augX
       prev_k    - 
       *K_prams   - N_layer x 1 - List of kernel parameters 
       layer_no  - 1 x 1
       *opt_prams - Parameters for each layer 
       
       
       Returns:
           
       Posterior mean(Y_pred)  - N* x 1
       Posterior cov           - N* x N*
       Optimised parms         - Depenedent on Kernel
   """


    inv_sigma  = np.linalg.inv(sigma_noise)
    #K(x*,x)
    k_xsx      =  kernel.K(Xsamp, Xpred)
    #K(x*,x)*[K(X,X) + whitenoise]^{-1}
    temp       = k_xsx.T @ inv_sigma 
    #f* | X, y, X* = K(x*,x)*[K(X,X) + whitenoise]^{-1}* Y
    post_mean1 = temp @ Y
#    Using GPy regression
    kernel = select_kernel()

    k_xsxs     = kernel.K(Xpred,Xpred)
    temp       = inv_sigma @ k_xsx
    post_cov1  = k_xsxs  - k_xsx.T @ temp
    post_vars1 = np.sqrt(post_cov1.diagonal())
    

    print(kernel.param_array)
#    using Gpy regression module we get a different posterior mean
    m          = GPy.models.GPRegression(Xsamp,Y,kernel)
    k_samp     = kernel.K(Xsamp).T
    m['Gaussian_noise.variance']          = 0.0001*np.amax(k_samp)
    post_mean, post_var = m.predict(Xpred)

    return post_mean1, post_var
# def combine_kernels(K1,K2):
#     """Creates the composite kernel"""
#     return K1 + K2

def select_kernel():
     """ Selects a new kernel at random 
     """
     var         = 1
     ls          = 2/math.pi
   
     kernel = GPy.kern.RBF(input_dim=2,variance=var,lengthscale=ls)
     return kernel
def layers(augX, layer_no):
    """ matplotlib.imagesc
        - Do GP regression + optimization
        - Make prediction for layer_output
    Inputs:
        
        augX               - New inputs +  Original inputs 
        Prev_Kernel        - Either composite or if first layer non-composite
        layer_no           - Layer number
        
           
        Outputs:
            
        * opt_prams        - The parameters of the kernel at different layers
        layer_output       - The conditional mean
        composite_kernel   - The sum of the kernels from all previous layers
        """
    kernel = select_kernel()
    Xsamp, Y,sigma_noise = sample_and_unroll(augX, kernel,layer_no)
#    [post_mean, post_cov,opt_prams, kernel] = regression(Xsamp, Y,augX,\
#                                                prev_kernel,layer_no)
   
    post_mean, post_cov = regression(Xsamp, Y,augX,\
                                                kernel,layer_no,sigma_noise)
    new_outputs = post_mean #this is generated from posterior conditional mean
    augX[:,0] = new_outputs[:,0] # equiv X = Y
    plt.figure(layer_no)
    plt.hold(False)
    name_of_plot = 'layer' + str(layer_no)
    plt.title(name_of_plot)
    plt.plot(augX[:,1],augX[:,0],'b-')
    name_of_plot = 'layer' + str(layer_no)
    file = '/Users/bradley/Documents/Aims_work/Miniproject1/My_code/src/DuvPlots'+'/'+name_of_plot +'.png'
    plt.savefig(file, format ='png')

#    return opt_prams, augX, kernel
    return augX
  
def sample_and_unroll(augX, kernel,layer_no):
    """ 
      - Selects a random sample of numbers from both the original inputs 
        augX[:,1] and previous outputs augX[:,0]. 
      - Then constructs y, which is sampled from : y ~ N(0, f(Xsamp))
        
        Inputs:
            
            augX            -  Updated inputs + original inputs
            *kernel_prams   -  So that we sample using the right composed kernel
            kernel          -  A combination of all previous kernels
        
        Ouputs:
            
            Xsamp          - A sampled subset of the augX
            Y               - Our outputs, given an Xsamp 
        
    """
    resolution   = 100
    lower_domain = np.amin(augX,axis=0)
    upper_domain = np.amax(augX,axis=0)

    D          = lower_domain.shape[0];
    N_1D       = int(np.ceil(np.power(resolution,1/D)))
    N          = np.power(N_1D,D)
    large_arr  = np.zeros((N_1D,D))
    large_arr2 = np.zeros((N_1D,D))

    for ii in range(D):
        large_arr[:,ii]  = np.linspace(lower_domain[ii],upper_domain[ii],N_1D)

    for ii in range(D):
        large_arr2[:,ii] = np.linspace(lower_domain[ii],upper_domain[ii],N_1D)

    large_arr   = np.repeat(large_arr,N_1D/2,axis=1)
    large_arr2  = np.repeat(large_arr2.T,N_1D/2,axis=0)
    coloum2     = np.ravel(large_arr)[:,None]
    coloum1     = np.ravel(large_arr2)[:,None]

    Xsamp       = np.concatenate((coloum1,coloum2),axis=1)

   
    # Generate Y randomly
    noise = 0.0001
    k     = kernel.K(Xsamp).T
    plt.imshow(k)
    name_of_plot = 'Plot_of_covariance_function' + str(layer_no)
    file = '/Users/bradley/Documents/Aims_work/Miniproject1/My_code/src/DuvPlots'+'/'+name_of_plot+'.png'
    plt.savefig(file,format= 'png')
    sigma_noise = k + noise*np.eye(Xsamp.shape[0])*np.amax(k)
    
    mu = np.zeros(np.size(Xsamp,axis=0))
    mu.flatten()
#   Y ~ f(X) + noise
    Y = np.random.multivariate_normal(mu,sigma_noise)[:,None]

    return Xsamp, Y,sigma_noise
    
def main():

#==============================================================================
#==============================================================================
# Initialisations
#==============================================================================
     N  = 1000
#     X  = np.random.uniform(-3,3,N)
     X = np.linspace(-1,1,N)
     X  = X[:,None]
     X0 = np.copy(X)
     augX = np.concatenate((X,X0),axis=1)
     runs = 20
     layer_output = np.array(np.shape(augX))
     np.random.seed (seed = 0 )

#==============================================================================
# Stacked DeepGP loop
#==============================================================================


     for ii in range(runs):
         layer_no = ii
         if ii > 0:
             layer_output = layers(layer_output, layer_no)
         else:
             layer_output = layers(augX, layer_no)
#==============================================================================



main()