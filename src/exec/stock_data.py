#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:45:36 2017

@author: bradley
"""

import pickle
import sys
import re
import numpy as np
sys.path.append("../")
import pandas as pd

path = '../../data/df_clean.pkl'
data = pd.read_pickle(path)
def get_stock(stock_no, if_2d = False):
    #data.info()
    #print(data.keys())
    keys = data.keys()
    
    
    #App_open = data['AAPL Open']
    
    #keys2 = App_open.keys()
    
    #calues = App_open.values
    p = re.compile(r'[A-Z]+\s+Open{1,20}')
    q = re.compile(r'[A-Z]+\s+Close{1,20}')
    stock_names_open  = []
    stock_names_close = []
    for key in keys:
        m  = p.findall(key)
        if m == []:
            continue
        else:
            stock_names_open.append(m)
    for key in keys:
        n  = q.findall(key)
        if n == []:
            continue
        else:
            stock_names_close.append(n)
    index1  = []
    index2  = []
    for stock in stock_names_open:
        index1.append(stock[0])
    for i in range(len(index1)):
        temp1    = index1[i]
        temp2    = temp1.replace(" Open","")
        stock_names_open[i] = temp2
    for stock in stock_names_close:
        index2.append(stock[0])
    for i in range(len(index2)):
        temp1    = index2[i]
        temp2    = temp1.replace(" Close","")
        stock_names_close[i] = temp2
   
    stock_data = pd.DataFrame(columns = stock_names_open)  
    stock_data1= pd.DataFrame(columns = stock_names_close)
    count = 0
    for stock in index1:
        temp1  = data[stock].values 
        stock_data[stock_names_open[count]]   = temp1
        count = count + 1
    count = 0
    for stock in index2:
        temp1  = data[stock].values 
        stock_data1[stock_names_close[count]]   = temp1
        count = count + 1   
    # to convert to NP array
    if if_2d:
        openstock  = stock_data[stock_names_open[stock_no]].as_matrix()
        closestock = stock_data1[stock_names_close[stock_no]].as_matrix()
        return closestock,openstock
    else:
        closestock = stock_data[stock_names_close[stock_no]].as_matrix()
        return closestock
    