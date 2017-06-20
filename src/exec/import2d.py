#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:54:13 2017

@author: bradley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:35:42 2017

@author: bradley
"""
import numpy as np
import math
import sys
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
        inputs         = np.linspace(0,1,size_data)
    elif option == 'stock':
      stock_no            = 13
      data                = sd.get_stock(stock_no = stock_no)
      data                = data[0:1000][:,None]
      size_data           = data.shape[0]
      inputs                = np.linspace(0,1,size_data)
    elif option == 'stock2d':
      stock_no            = 13
      data,input2         = sd.get_stock(stock_no,if_2d = True)
      data                = data[0:1000][:,None]
      size_data           = data.shape[0]
      inputs1             = np.linspace(0,1,size_data)
      inputs2             = input2[0:1000][:,None]
      inputs              = np.column_stack((inputs1,inputs2))
    elif option == '2d':
      samples   = 500
      inputs1   = np.linspace(-math.pi,math.pi, samples)[:,None]
      inputs2   = np.linspace(-math.pi,math.pi, samples)[:,None]
      inputs    = np.column_stack((inputs1,inputs2))
      data      = np.sin(inputs[:,0:1]) * np.sin(inputs[:,1:2]) +np.random.randn(samples,1)*0.05
      size_data = data.shape[0]
    
    if predict_points:
        inputs = inputs[0:size_data,:]
        return inputs, data,size_data
    else:
        no_points = int(split*size_data)
        data      = data[0:no_points,:]
        assert np.shape(data)[0] == no_points
        inputs    = inputs[:no_points,:]
        return inputs, data, no_points