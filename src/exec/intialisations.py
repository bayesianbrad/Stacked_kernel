#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:43:54 2017

@author: bradley
"""
# To do, split will eventually be timeds by no_points and that number will
# feed through to import data, which will then separate into the appropriate
# testing and training sets. 

def get_initial(split,option,no_dims, is2d = False):
    """This module prepares intialisations"""
    if is2d:
        import import2d as impdata
    else:
        import importdata as impdata
    augtimes, data,no_points = impdata.import_data(split,option,no_dims)
    # To eventually be determined by the autocorrelation fucntion
    data_train_key          = option + '_train'
    data_val_key            = option + '_val'
    data_keys               = [data_train_key, data_val_key]
    input_data              = dict()
    input_data['train']     = augtimes
    input_data[data_keys[0]]= data
    
    return input_data,data_keys,no_points
