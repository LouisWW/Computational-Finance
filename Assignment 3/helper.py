#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh import FdMesh


# FD_mesh(S_max,dS,T_max,dT)
Grid = FdMesh(1, 0.2 , 1, 0.2)

first = [{'value': -1, 'offset': 1}, {'value': 1, 'offset': -1}]
second = [{'value': 2, 'offset': 0}, {'value': 1, 'offset': -1}, {'value': 1, 'offset': 1}]

Grid.tri_diag_matrix_func(3, 4, 5, 6, first, printing=True)
Grid.tri_diag_matrix_func(3, 4, 5, 6, second, printing=True)




