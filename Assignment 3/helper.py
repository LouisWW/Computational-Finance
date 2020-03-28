#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh_2 import FdMesh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import tqdm

import numpy as np

# To test
def test(s_min=0,s_max=200, ds=1, t_max=1, dt=0.001, S0=100, K=100, r=0.04,sigma=0.3, option='call', fm_type='crank-nicolson'):
    '''
    Test the pde
    :return:
    '''


    Grid = FdMesh(s_min, s_max, ds, t_max, dt, S0, K, r, sigma, option, fm_type)
    Grid.run()

    l1 = []
    l2 = []
    s = range(10, 195, 5)
    for i in s:
        l1.append(Grid.greek(i))
        l2.append(Grid.delta)

    plt.plot(s, l1, label='PDE', color='blue')
    plt.plot(s, l2, '--', label='Black-Scholes', color='red')

    plt.legend()
    plt.xlabel('Stock price', fontsize=17)
    plt.ylabel('Delta', fontsize=17)
    plt.show()
    # #print(Grid)
    #
    # X, Y = np.meshgrid(Grid.stock_prices,Grid.t)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    #
    #
    # # Plot the surface.
    # ax.plot_surface(X.T, Y.T,Grid.grid, alpha=1, rstride=1, cstride=1, cmap=cm.winter, linewidth=0.5, antialiased=True,
    #                 zorder=0.5)
    #
    # ax.set_xlabel('S',fontsize=17)
    # ax.set_ylabel('T',fontsize=17)
    # ax.set_zlabel('option price',fontsize=17)
    # plt.show()



def diff_S0(s_min=0,s_max=300,ds=1,t_max=1,dt=(0.0001,0.01),S0=(100,110,120),K=110,r=0.04,sigma=0.3,option='call'):
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


def convergence(s_min=0, s_max=100, t_max=1, S0=50, K=60, r=0.04, sigma=0.2, option='call', fm_type='forward'):

    ds_list = np.linspace(5, 0.5, 10)
    # dt_list = np.array([0.5 ** i for i in range(1, 11)])
    dt_list = np.linspace(0.1, 10e-4, 10)

    comp_price_grid = np.zeros((len(ds_list), len(dt_list)))

    for i, ds in tqdm.tqdm(enumerate(ds_list)):
        for j, dt in enumerate(dt_list):
            Grid = FdMesh(s_min, s_max, ds, t_max, dt, S0, K, r, sigma, option, fm_type)
            FM_price = Grid.run()
            if abs(FM_price) > 5 or np.isnan(FM_price):
                FM_price = 5
            comp_price_grid[j][i] = FM_price

    fig = plt.figure()
    ax = Axes3D(fig)

    X, Y = np.meshgrid(ds_list, dt_list)

    # Plot the surface.
    ax.plot_surface(X.T, Y.T, comp_price_grid, alpha=1, rstride=1, cstride=1, cmap=cm.winter, linewidth=0.5, antialiased=True,
                    zorder=0.5)

    ax.set_ylabel('dt', fontsize=17)
    ax.set_xlabel('ds', fontsize=17)
    ax.set_zlabel('option price', fontsize=17)
    plt.show()



