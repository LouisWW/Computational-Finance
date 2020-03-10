#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from FD_mesh import FD_mesh


# FD_mesh(S_max,dS,T_max,dT)
Grid = FD_mesh(10,1,10,1)

print(Grid)


