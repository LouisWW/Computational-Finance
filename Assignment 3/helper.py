#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh_2 import FdMesh
import matplotlib.pyplot as plt

Grid = FdMesh(s_min=0,
              s_max=100,
              ds=1,
              t_max=1,
              dt=0.0005,
              S0=50,
              K=60,
              r=0.05,
              sigma=0.2,
              option='put',
              fm_type='forward')


Grid.run()
print(Grid)

# first = [{'value': -1, 'offset': 1}, {'value': 1, 'offset': -1}]
# second = [{'value': 2, 'offset': 0}, {'value': 1, 'offset': -1}, {'value': 1, 'offset': 1}]
#
# Grid.tri_diag_matrix_func(3, 4, 5, 6, first, printing=True)
# Grid.tri_diag_matrix_func(3, 4, 5, 6, second, printing=True)
