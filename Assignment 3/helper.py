#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh_2 import FdMesh
import matplotlib.pyplot as plt

import numpy as np

# To test
def test(s_min=0,s_max=100,ds=1,t_max=1,dt=0.001,S0=50,K=60,r=0.05,sigma=0.2,option='put',fm_type='forward'):
    '''
    Test the pde
    :return:
    '''
    Grid = FdMesh(s_min,s_max,ds,t_max,dt,S0,K,r,sigma,option,fm_type)
    Grid.run()



def diff_S0(s_min=0,s_max=200,ds=1,t_max=1,dt=(0.0001,0.01),S0=(100,110,120),K=110,r=0.04,sigma=0.3,option='call'):
    '''
    Compares the Explicit with the Crank-nicolson method using 3 different S0.
    '''

    S0_1,S0_2,S0_3=S0
    dt_1,dt_2=dt

    print("General paramters are K={}, r={}, vol={}, option_type={},".format(K,r,sigma,option))
    print("S0={}".format(S0_1))
    Grid_1f = FdMesh(s_min,s_max,ds,t_max,dt_1,S0_1,K,r,sigma,option,fm_type='forward')
    Grid_1f.run()
    Grid_1c = FdMesh(s_min,s_max,ds,t_max,dt_2,S0_1,K,r,sigma,option,fm_type='crank-nicolson')
    Grid_1c.run()

    print("------------------------------------------")
    print("S0={}".format(S0_2))

    Grid_2f = FdMesh(s_min, s_max,ds, t_max, dt_1, S0_2, K, r, sigma, option, fm_type='forward')
    Grid_2f.run()
    Grid_2c = FdMesh(s_min, s_max, ds, t_max, dt_2, S0_2, K, r, sigma, option, fm_type='crank-nicolson')
    Grid_2c.run()

    print("------------------------------------------")
    print("S0={}".format(S0_3))
    Grid_3f = FdMesh(s_min,s_max,ds,t_max,dt_1,S0_3,K,r,sigma,option,fm_type='forward')
    Grid_3f.run()
    Grid_3c = FdMesh(s_min,s_max,ds,t_max,dt_2,S0_3,K,r,sigma,option,fm_type='crank-nicolson')
    Grid_3c.run()


def convergence(s_min=0,s_max=200,t_max=1,S0=50,K=60,r=0.04,sigma=0.2,option='call'):
    pass
