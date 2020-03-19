#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh_2 import FdMesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

# To test
Grid = FdMesh(s_min=0,
              s_max=100,
              ds=1,
              t_max=1,
              dt=0.001,
              S0=50,
              K=60,
              r=0.05,
              sigma=0.2,
              option='put',
              fm_type='forward')
Grid.run()

# Test examples given
# r= 4% sigma=30 S0=100 K=110 T=1
Grid_1f = FdMesh(s_min=0,s_max=150,ds=1,t_max=1,dt=0.001,S0=100,K=110,r=0.04,sigma=0.3,option='call',fm_type='forward')
Grid_1f.run()
Grid_1c = FdMesh(s_min=0,s_max=150,ds=1,t_max=1,dt=0.001,S0=100,K=110,r=0.04,sigma=0.3,option='call',fm_type='crank-nicolson')
Grid_1c.run()

#X,Y=np.meshgrid(Grid.t,Grid.stock_prices)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(Y,X,Grid.grid, 50, cmap='binary')

#plt.show()
