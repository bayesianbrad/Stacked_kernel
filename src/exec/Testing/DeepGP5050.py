#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:26:36 2017

@author: bradley gram-hansen
1input dimension:  Deep GP for testing
Pseudo diagram:

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
the posterior mean and the orginba; data , is around the 0 axis, i.e
the abs(input data - posterior mean of layer) < 1 - white noise


To do:
Add auto-correlation between the posterior mean and 

For testing: 
    
Within fucntions * indicates maybe not necessary
"""


import GPy
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import pandas as pd 
np.random.seed (seed = 0 )
plt.style.use('ggplot')

try:
    plt.close("all")
except:
    pass
def import_data(no_points,option,predict_points=False):
    """Depending on the 'option' specified, will load which ever
      data set it has been programmed for. Currently financial and sunspots
      
      Input:
          
          option        -  string of data set name
        
      Output:
          
          X data for training data
          Y data for training data
    """
    import sys
    sys.path.append("../")
    import scipy.io as sio
    
    if option == 'financial':
        # data = dict()
        # filename       = 'finPredProb.mat'
        # financial_data = sio.loadmat('../../data/' + filename)
        # data['ytrain']      = financial_data['tte'][:,0]
        # data['ytest']       = financial_data['ttr'][:,0]
        # time_train          = np.shape(financial_data['ttr'][:,0])
        # time_test           = np.shape(financial_data['tte'][:,0])
        # data['time_train']  = np.arange(0,time_train,1)
        # data['time_test']   = np.arange(0,time_test,1)
        return data
        # clear from memeory excess data
        financial_data = None
    elif option == 'mcglass':
        data = dict()
        filename       = 'mg.mat'
        mcglass_data   = sio.loadmat('../../data/' + filename)
        data['ytrain']      = mcglass_data['t_tr']
        data['ytest']       = mcglass_data['t_te']
        time_train          = np.linspace(0,1,data['ytrain'].shape[0])
        time_test           = np.linspace(0,1,data['ytest'].shape[0])
        data['ttrain']  = np.concatenate((time_train[:,None],time_train[:,None]),axis=1)
        data['ttest']   = time_test[:,None]
        return data
    elif option == 'sunspot':
        filename     = 'sunspots.txt'
        filename     = '../../data/' + filename
        sunspot_data = np.loadtxt(filename, dtype='float')
        size_data     = np.size(sunspot_data)
       

        # months go form 0 - 1, instead of 0 - 2899
        #time             = np.linspace(0,1,2899)
        time             = np.linspace(0,1,size_data)  
        if predict_points:
            time_predict = time[0:size_data]
            time_predict = time[:,None]
            sunspot_data = sunspot_data[:,None]
            return time_predict, sunspot_data

        else:
            sunspot_data   = sunspot_data[0:no_points]
            sunspot_data   = sunspot_data[:,None]
            time           = time[:no_points]
            time           = time[:,None]
            time_input     = np.copy(time)
            # Ensure that input goes through to every layer 
            augData        = np.concatenate((time,time_input),axis=1)
            return augData, sunspot_data

def normalise_data(data):
    """Takes training and test data and normalises it, to ensure zero mean
       and unit variance. 
       
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

def unit_vector(data):
    """Ensures data is between -1 and 1"""
    norm     = sum(np.dot(data.T,data))
    scalar   = 1 / math.sqrt(norm)
    norm     = data*scalar
    try:
        assert sum(norm**2) > 0.9999999 and sum(norm**2) <1.0000001
    except:
        print(sum(norm**2))
    return norm

def split_data_for_train(augtimes, data,split,option):
    """Splits the training data into split% training and 1-split% validation
       Assumes augtime and data have same row lengths. 
        Inputs: 
        
            augtime      - N x M array (for sunspot N = 2899, M =2 (new output+ 
                                                               orig_input)) 
            
            data         - N x 1 array 
            
        Outputs:
            
            inputs_dict   - A dictionary containing the following key-values:
                key: augtrain             - split%  of  augtime
                key: augval               - split%  of  augtime
                key: option+'_train'      - split%  of  data
                key: option+'_val'        - split%  of  data
            
        """
    no_elements = augtimes.shape[0]
    # As in the 50-50 DeepGP training and validation set must be equal
    # we require that the shape be of the inout and output data be even
    # Proposal: Remove last row in array if odd, else keep the same:
    
    if no_elements%2 != 0:
        augtimes = augtimes[0:-1]
        data     = data[0:-1]
    else:       
        pass
        
        
    # Deals with odd and even sized augtimes
    # I leave data_train and data_val in the for loop, in case the model is
    # extended for multiple outputs. 
    if split > 0 and split <1:
        split        = -no_elements + int(split*no_elements)
        augtimes_train =  np.zeros((no_elements+split,augtimes.shape[1]))
        augtimes_val   =  np.zeros((-split,augtimes.shape[1]))
        data_train     =  np.zeros((no_elements+split,1))
        data_val       =  np.zeros((-split,1))
        for ii in range(augtimes.shape[1]):
            augtimes_train[:,ii] = augtimes[:split,ii]
            augtimes_val[:,ii]   = augtimes[split:,ii]
            data_train[:,0]      = data[:split,0]
            data_val[:,0]        = data[split:,0]
    elif split == 1:
        split        = -no_elements + int(split*no_elements)
        augtimes_train =  np.zeros((no_elements,augtimes.shape[1]))
        augtimes_val   =  None
        data_train     =  np.zeros((no_elements,1))
        data_val       =  None
        for ii in range(augtimes.shape[1]):
            augtimes_train[:,ii] = augtimes[:,ii]
            data_train[:,0]      = data[:,0]




        inputs_dict = dict()
        data_train_key     = option + '_train'
        data_val_key       = option + '_val'
        data_keys          = [data_train_key, data_val_key]
        inputs_dict['augtrain']   = augtimes_train
        inputs_dict['augval']     = augtimes_val
        inputs_dict[data_keys[0]]     = data_train
        inputs_dict[data_keys[1]]     = data_val
    else:
        print("split must be between 0 and 1")

     

     

    return inputs_dict, data_keys    

def select_kernel():
     """ Selects a new kernel at random 
     """
     # kernel_list = ['RBF','Matern32','Matern52','MLP']
     # kernel_number = np.random.randint(0,3)
     kernel_list   = ['RBF']
     kernel_number = np.random.randint(0,1)
     # getattr allows us to get the related method from the class 
     kernel = getattr(GPy.kern,kernel_list[kernel_number])(input_dim=2,\
                             variance=math.sqrt(math.pi/2))
     # test
     print(kernel_list[kernel_number])
     return kernel

def combine_kernels(K1,K2):
    """Creates the composite kernel"""
    return K1 + K2

def fin_intialisations(no_points,option):
    """This module will take the financial data and 
       separate it as follows: """
def simple_intialisations(no_points,option):
    X = np.linspace(-3.,3.,no_points)
    X = X[:,None]
#    Y = 0.5 * (np.sign(X) + 1) 
#    X0   = np.copy(X)
#    augX = np.concatenate((X,X0),axis=1)
    ####################################
    # For 1- dim output and input
    augX  = X
    Y     = np.sin(X) + np.random.randn(no_points,1)

  
    Y           = normalise_data(Y)
    Y           = unit_vector(Y)
    init_kernel = select_kernel()

    runs        = 20
    split       = 1

    input_data,data_keys  =  split_data_for_train(augX,Y,split,option)  

    return input_data,data_keys, init_kernel, runs


def sunspots_intialisations(no_points,option):
    """This function takes sunspot data and creates 
       training and testing sets and intialises the kernel 
       and run number
    """
    
    augtimes, sunspotdata = import_data(no_points,option)
    sunspotdata           = normalise_data(sunspotdata)
    # Make unit vector
    sunspotdata           = unit_vector(sunspotdata)
    init_kernel           = select_kernel()
    
    # To eventually be determined by the autocorrelation fucntion
    runs                  = 5
    # Separate into training and validation set 
    split                 =  1
    input_data,data_keys  =  split_data_for_train(augtimes,sunspotdata,split,option)  
    
    return input_data, data_keys, init_kernel, runs


def mcglass_intialisations(no_points, option):
    """This function takes the mcglass data and returns training 
       and validation datasets"""

    data              = import_data(no_points,option)
    data['ytrain']    = normalise_data(data['ytrain'])
    # Make unit vector
    data['ytrain']        = unit_vector(data['ytrain'])
    init_kernel           = select_kernel()

    #To eventually be determined by the autocorrelation fucntion
    runs                  =  20
    # Separate into training and validation set 
    split                 =  0.5
    input_data,data_keys  =  split_data_for_train(data['ttrain'],data['ytrain'],split,option)  
    
    return input_data, data_keys, init_kernel, runs 

def error(prediction, groundtruth):
  """Calculates the RMSE"""
  groundtruth = groundtruth[:prediction.shape[0]]
  assert np.size(prediction) == np.size(groundtruth)
  diff_square = (prediction - groundtruth)**2
  error       =  np.mean(diff_square)
  rmse        =  math.sqrt(error)
  return rmse



def regression(Xtrain,Ytrain,Xval,Yval,prev_k, layer_no, no_points,option):
    """GP regression module - uses GPy
       - Generates a new kernel, made from the previous
       kernel. 
       - Carries out GP regression + optimisation
       - Does the prediction for the output of the layer
       
       Inputs:
           
       Xtrain     - N  x 2 - split% training data
                           
       Ytrain     - N  x 1 - split% output from training data
                          
       Xval       - N* x 2 - Future times to predict
       
       Yval       - N* x 1 - So that it can be passed to plotting() for 
                            verification
                   
       prev_k     - Linear combination of the previous kernels
       
       layer_no   - 1 x 1
       *opt_prams - Parameters for each layer 
       
       
       Returns:
           
       Posterior mean(Y_pred)  - N* x 1
       Posterior cov           - N* x N*
       Optimised parms         - Depenedent on Kernel
   """
#    if layer_no >0:
#        k     = select_kernel()
#        k_add = combine_kernels(prev_k,k)
#    else:
#        k_add = prev_k
    k_add = prev_k
    m = GPy.models.GPRegression(Xtrain, Ytrain, k_add) 
    m.optimize(max_iters = 1000)
    
#==============================================================================
#     hand mean and cov
#    Xsamp = Xtrain 
#    noise = 0.0001
#    k     = select_kernel()
#    kernel= k.K(Xsamp,).T 
#    sigma_noise = kernel + noise*np.eye(Xsamp.shape[0])*np.amax(kernel)
#    inv_sigma  = np.linalg.inv(sigma_noise)    
#==============================================================================


    # ****************************************
    # if layer_no > 0:
    #   k_add = select_kernel()
    # else:
    #   k_add = prev_k

    # m = GPy.models.GPRegression(Xtrain, Ytrain, k_add) 
    # # if count <= 1:
    # #   m.optimize('scg', max_iters = 300)
    # #   # add prev kernel to optimised kernel to generate new kernel
    # #   # add one  to count
    # # else:
    # #   # count = 0
    # #   # kern  =select_kernel()
    # #   m.optimize('scg', max_iters = 300)
    # #   # k_add =  prev_kern + kern 
    # m.optimize('scg', max_iters = 600)
    # if layer_no > 0:
    #   k_add = k_add + prev_k
    #   m = GPy.models.GPRegression(Xtrain, Ytrain, k_add)

    # *****************************************



    # for full covariance return and to use a different kernel for prediction
    # use m.predict(Xnew, full_cov=True, kern = < >)
   




#•#•#•#•#•#•#•#• For testing option = 'simple'#•#•#•#•#•#•#•#•
    Xtest1  = np.linspace(-1.5,0,10)
    Xtest1  = Xtest1[:,None]
    Xtest2  = np.linspace(0,1.5,10)
    Xtest2  = Xtest2[:,None]
    Xtest   = np.concatenate((Xtest1,Xtest2),axis=0)
    post_mean, post_var = m.predict(Xtest)
    #K(x*,x)


#   post_vars1 = np.sqrt(post_cov1.diagonal())
    if layer_no == 1:
        m.plot()
        plt.plot(Xtest.flatten(),post_mean.flatten(),'r-')
        plt.show(block=True) 
    if layer_no == 5:
        m.plot()
        plt.plot(Xtest.flatten(),post_mean.flatten(),'r-')
        plt.show(block=True) 
    if layer_no == 15:
        m.plot()
        plt.plot(Xtest.flatten(),post_mean.flatten(),'r-')
        plt.show(block=True) 
    if layer_no > 40:
        m.plot()
        plt.show(block=True) 
# •#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•#•
# Prediction
    # Xtest, Ytest        = import_data(no_points,option,predict_points=True)
    # Ytest               = normalise_data(Ytest)
    # Ytest               = unit_vector(Ytest)
    # post_mean, post_var = m.predict(Xtest)
    # post_mean           = normalise_data(post_mean)
    # post_mean           = unit_vector(post_mean)
    # assert post_var.shape[1] == 1
    # post_var            = normalise_data(post_var)
    # post_var            = unit_vector(post_var)

    # plotting(Xtrain,Ytrain,Xtest,Ytest,post_mean,post_var,layer_no)
    # to Implement: save_output(Xtrain,post_var,post_mean)



#    opt_prams[layer_no] = [{'variance':k_add['.*var']},{'lengthscale':\
#              k_add['.*lengthscale']}]

    return post_mean, post_var, k_add, Xtest

def layers(input_data, prev_kernel,layer_no,no_points,data_keys,option):
    """ 
        - Do GP regression + optimization
        - Make prediction for layer_output
    Inputs:
        
        input_data         - Dictionary containing all training and validation
                             data arrays for X inputs and Y outputs
        prev_Kernel        - Either composite or if first layer non-composite
        layer_no           - Layer number
        option             - String of whatever dataset was choosen
        data_keys          - A list of the keys for input_data keys for outputs
        
           
        Outputs:
            
        * opt_prams        - The parameters of the kernel at different layers
        layer_output       - The conditional mean
        composite_kernel   - The sum of the kernels from all previous layers
     """
    Xtrain = input_data['augtrain']
    Xval   = input_data['augval']
    Ytrain = input_data[data_keys[0]]
    Yval   = input_data[data_keys[1]]
#    [post_mean, post_cov,opt_prams, kernel] = regression(Xsamp, Y,augX,\
#                                                prev_kernel,layer_no)
    post_mean, post_cov, kernel,Xval = regression(Xtrain, Ytrain,Xval,Yval,\
                                                prev_kernel,layer_no,no_points,\
                                                option)
    #this is the posterior conditional mean "prediction for yval"
    
    temp1 = Xval[0:no_points,0]
    temp1 = temp1[:,None]
    temp2 = post_mean[0:no_points,0]
    temp2 = temp2[:,None]
    assert temp1.shape[0] == temp2.shape[0]
    input_data['aug_train'] = temp2
#    input_data['augtrain'] = np.concatenate((temp1,temp2),axis = 1)
    
        
#    if layer_no >0:
#        input_data['augtrain'][Xtrain.shape[0] - Xval.shape[0]:,0]   = new_outputs[0,:]
#    else:
#        input_data['augtrain']   = np.concatenate((Xtrain,new_outputs),axis=0)
#        input_data[data_keys[0]] = np.concatenate((Ytrain,Yval),axis=0)

#    return opt_prams, augX, kernel
    return input_data, kernel

def plotting(Xtrain,Ytrain,Xval,Yval,post_mean,post_var,layer_no):
    """For plotting the graphs to verify the confidence in the prediction of
       our training data. Saves plots in ../../Plots. Takes the model, m
       and uses GPys inbuilt plotting. 
       Inputs:
           
           mu               - posterior mean (post_mean)
                              augX[:,1] original inputs
           
            Y               - The given output for the training or test data
            
        post_cov            - The diagonal of cov(x*,x*) for plotting confidence
        
        *kernel             - The kernel up to this layer
        
        Output:
            
         plots               - Saved as pdf to given file name 
   """
    # For option = 'simple'
    # if Xval:

    #     s  = post_var.flatten()
    #     mu = post_mean.flatten()
    #     plt.figure(1)
    #     plt.clf()
    #     plt.hold(True)
    #     plt.plot(Xtrain[:,0].flatten(),Ytrain.flatten(), 'r-', ms=20)
    #     plt.plot(Xtrain[:,0].flatten(), Yval[:,0].flatten(), 'b-')
    #     plt.gca().fill_between(Xtrain[:,0].flatten(), mu-2*s, mu+2*s, color="#977ffa")
    #     plt.plot(Xtrain[:,0].flatten(), mu, 'g-', lw=2)
    #     plt.title("This is layer"+str(layer_no))
    #     name_of_plot = 'Prediction in  layer' + str(layer_no)
    #     file = '/Users/bradley/Documents/Aims_work/Miniproject1/My_code/src/Plots'+'/'+name_of_plot
    #     plt.savefig(file, format = 'png')
    # else:
    s  = post_var.flatten()
    mu = post_mean.flatten()
    plt.figure(1)
    plt.clf()
    plt.hold(True)
    plt.plot(Xtrain[:,0].flatten(),Yval[0:Xtrain.shape[0],0].flatten(), 'r-')
    plt.plot(Xval[Xtrain.shape[0]:,0].flatten(), Yval[Xtrain.shape[0]:,0].flatten(), 'y-')
    plt.gca().fill_between(Xval[:,0].flatten(), mu-2*s, mu+2*s, color="#977ffa")
    plt.plot(Xval[:,0].flatten(), mu, 'g-', lw=2)
    plt.title("This is layer"+str(layer_no))
    name_of_plot = 'Prediction in  layer' + str(layer_no)
    file = '/Users/bradley/Documents/Aims_work/Miniproject1/My_code/src/Plots'+'/'+name_of_plot
    plt.savefig(file, format = 'png')


def saveoutput(Xtrain,post_var, post_mean):
  #To complete
    df = pd.DataFrame(Xtrain)
    df.to_csv("file_path.csv", header=None)

def main():

#==============================================================================
#==============================================================================
# Initialisations
#==============================================================================

     # option   = 'sunspot'
     #option = 'financial'
     #option = 'mcglass'
     option = 'simple'
     no_points = 20
     if option == 'sunspot':
        # Will update code later to get rid of augmenting as MAYBE not needed
        input_data,data_keys,init_kernel,runs = sunspots_intialisations(no_points,option)
        # Xtrain     = input_data['augtrain']
        # temp       = Xtrain[:,0]
        # temp       = temp[:,None]
        # input_data['augtrain'] = temp 
        # Xval     = input_data['augval']
        # temp       = Xval[:,0]
        # temp       = temp[:,None]
        # input_data['augval'] = temp
     elif option == 'financial':
         print("Not yet implemented")
     elif option == 'mcglass':
        input_data, data_keys, init_kernel, runs = mcglass_intialisations(no_points,option)
     elif option == 'simple':
        input_data, data_keys, init_kernel, runs = simple_intialisations(no_points,option)
        # Xtrain     = input_data['augtrain']
        # temp       = Xtrain[:,0]
        # temp       = temp[:,None]
        # input_data['augtrain'] = temp 
        # Xval     = input_data['augval']
        # temp       = Xval[:,0]
        # temp       = temp[:,None]
        # input_data['augval'] = temp

    

#==============================================================================
# Stacked DeepGP loop
#==============================================================================
#Currently not using the opt prams, will have to see how GPy handles kernel
#passing 
#NOTE: Layer_output contains the new outputs coloumn '0' and the original inputs
#      will always be in coloum '1' - python indexing.

     for ii in range(runs):
         layer_no = ii
         if ii > 0:
             input_data, kernel = layers(input_data,kernel,layer_no,no_points,data_keys,option)
             print(kernel)
             print("This is layer " , ii+1)
             # print(input_data['augtrain'][:,:])
             # print(input_data['augval'][:,:])
             # print(input_data[data_keys[0]])
         else:
             input_data, kernel = layers(input_data, init_kernel,layer_no,no_points,data_keys,option)
             print(kernel)
             print("This is layer " , ii+1)
             # print(input_data['augtrain'][:,:])
             # print(input_data['augval'][:,:])
             # print(input_data[data_keys[0]])


main()

