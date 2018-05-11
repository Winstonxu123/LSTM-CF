#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:57:51 2017

@author: xu
"""

import re
import numpy as np
import matplotlib.pyplot as plt

def str2num(str_list,output_type):
    return [output_type(x) for x in str_list]

if '__main__' == __name__:
    log_file='/home/xu/myWorkspace/LSTM-CF/Demo/log/train3_2018-03-06_22:47:20.log'
    pattern_iter=re.compile(r'214\]\s+Iteration\s+([\d]+)')
    #pattern_loss = re.compile(r"Train net output #0: loss[\s=]{1,3}([\d\.]+)")
    pattern_loss=re.compile(r'\, loss[\s=]{1,3}([\d\.]+)')
    with open(log_file, 'r') as f:
        lines = f.read()
        itrs = pattern_iter.findall(lines)
        loss = pattern_loss.findall(lines)
        
     
        itrs = np.array(str2num(itrs, int))
        losses = np.array(str2num(loss, float))
        plt.figure()
        plt.plot(itrs, losses[:-1])
        plt.title("Loss vs Iters")
        plt.show()
