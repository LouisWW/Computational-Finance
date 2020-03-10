#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh import FdMesh


# FD_mesh(S_max,dS,T_max,dT)
Grid = FdMesh(0,1,10,1)


Grid.tri_diag_matrix(3,4,5,6)




