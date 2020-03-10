#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import numpy as np


class FdMesh:

    def __init__(self, s_max, ds, t_max, dt):
        self.s_max = s_max
        self.ds = ds
        self.t_max = t_max
        self.dt = dt

        # To make the grid from 0 to T_max/S_max
        self.grid = np.zeros((self.t_max+1,self.s_max+1))


    def forward_approximation(self):
        pass

    def __str__(self):

        self.str="\n\n Finite Difference mesh \n\n"
        # To make sure the coordinate (0,0) is at the bottom left
        for i in range(self.grid.shape[1]-1,-1,-1):
            for j in range(self.grid.shape[0]):

                self.str += str(self.grid[j][i])
                self.str += "  "
            self.str += "\n"

        self.str += "\n\n\n\n"

        return self.str
