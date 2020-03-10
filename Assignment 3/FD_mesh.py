#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import numpy as np

class FD_mesh:

    def __init__(self, S_max,dS,T_max,dT):
        self.S_max = S_max
        self.dS = dS
        self.T_max = T_max
        self.dT = dT

        # To make the grid from 0 to T_max/S_max
        self.grid = np.zeros((self.T_max+1,self.S_max+1))


    def forward_approximation(self):
        pass

    def __str__(self):

        self.str="\n\n Finite Difference mesh \n\n"
        for i in range(self.S_max):
            for j in range(self.T_max):

                self.str += str(self.grid[i][j])
                self.str += "  "
            self.str += "\n"

        self.str += "\n\n\n\n"

        return self.str
