#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:35:42 2017

@author: bradley
"""
import numpy as np
import sys
import weierstrass as ws
sys.path.append("../")
import scipy.io as sio
import stock_data as sd

def import_data(split,option,no_dims,predict_points=False,):
    """Depending on the 'option' specified, will load which ever
      data set it has been programmed for. By design does augmented 
      input, unless single_input = True is passed.
      
      Input:
          split         -  size of training data
          option        -  string of data set name
          
          predicted_points - returns Xtest and Ytest for prediction
                              plotting
          single_input     - returns single input, not augmented
        
      Output:
          
          X data for training data - N X ? - ? either 1 or 2
          Y data for training data - N x ! - Multiple outputs not implemented.
    """
    if option == 'financial':
        filename     = 'finPredProb.mat'
        all_data     = sio.loadmat('../../data/' + filename)
        data         = all_data['xtr']
        data         = data[0:1000,0][:,None]
        size_data    = data.shape[0]
        time         = np.linspace(0,1,size_data)

    elif option == 'mcglass':
        filename     = 'mg.mat'
        all_data     = sio.loadmat('../../data/' + filename)
        data         = all_data['t_te'][0:1000]
        data         = data[:,0][:,None]
        size_data    = data.shape[0]
        time         = np.linspace(0,1,size_data)

    elif option == 'sunspot':
        filename     = 'sunspots.txt'
        filename     = '../../data/' + filename
        data         = np.loadtxt(filename, dtype='float')
        data         = data[:,None]
        size_data    = np.size(data)
        #months go form 0 - 1, instead of 0 - 2899
        time         = np.linspace(0,1,size_data)  
    
    elif option == 'simple':
        samples      = 50
        time         = np.linspace(-3,3,samples)
        data         = np.sin(time[:,None]) + np.random.rand(samples,1)
        size_data    = np.size(data)
        
    elif option == 'weierstrass':
      samples        = 1000
      time           = np.linspace(0,1,samples)
      data           = ws.weierstrass(time,i=3, j=6)[:,None]
      size_data      = np.size(data)
     
    elif option == 'heartbeat':
      filename       = 'hr_ts.txt'
      data           = np.loadtxt('../../data/' + filename)
      data                = data[:,None]
      size_data           = data.shape[0]
      time                = np.linspace(0,1,size_data)
    elif option == 'stock':
      stock_num           = 1
      data                = sd.get_stock(stock_num)
      data                = data[0:1000][:,None]
      size_data           = data.shape[0]
      time                = np.linspace(0,1,size_data)
      
    if predict_points:
        time = time[0:size_data][:,None]
        if no_dims == 1:
            return time, data,size_data
        else:
            time_init = np.copy(time)
            time      = np.concatenate((time,time_init),axis=1)
            return time, data,size_data
            
    else:
        no_points = int(split*size_data)
        data      = data[0:no_points]
        assert np.shape(data)[0] == no_points
        time      = time[:no_points][:,None]
        if no_dims == 1:
            return time, data, no_points
        else:
            time_init  = np.copy(time)
            time       = np.concatenate((time,time_init),axis=1)
            return time, data, no_points