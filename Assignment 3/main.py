#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import helper as helper

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np



'''
To test the finite method changing the different parameters
default parameters:
s_min=0,s_max=100,ds=1,t_max=1,dt=0.001,S0=50,K=60,r=0.05,sigma=0.2,option='put' or 'call',fm_type='forward' or 'crank-nicolson'
'''
helper.test()


"""
Compare the FTCS with the Cran-nicolson method using different stock price S0
default parameter:
s_min=0,s_max=200,ds=1,t_max=1,dt=(0.00001,0.01),S0=(100,110,120),K=110,r=0.04,sigma=0.3,option='call
"""
#helper.diff_S0()


#helper.convergence(fm_type='crank-nicolson')

