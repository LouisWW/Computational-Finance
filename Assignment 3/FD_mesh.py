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
        assert s_max !=0 , "s_max needs to be greater than 0 !!"
        self.s_max = s_max
        self.ds = ds
        self.t_max = t_max
        self.dt = dt

        # To make the grid from 0 to T_max/S_max
        self.grid = np.zeros((self.t_max+1,self.s_max+1))


    def forward_approximation(self):
        pass

    def tri_diag_matrix(self,k1,k2,k3,k4):
        # the size of the matrix depends only on the number of discrete stock prices
        self.tri_diag_matrix=np.zeros((self.grid.shape[1],self.grid.shape[1]))

        self.tri_diag_matrix[0][0] =k1
        self.tri_diag_matrix[1][0] = k2
        self.tri_diag_matrix[-2][-1] = k3
        self.tri_diag_matrix[-1][-1] = k4

        for i in range(0,self.tri_diag_matrix.shape[1]-2):
            self.tri_diag_matrix[i][i+1]=1
            self.tri_diag_matrix[i+2][i+1] = -1

        # Can be deleted once it is defined as stable with all sizes of matrix
        sstr =""
        for i in range(self.grid.shape[1]):
            for j in range(self.grid.shape[1]):

                sstr += str(self.tri_diag_matrix[j][i])
                sstr += "  "
            sstr += "\n"

        sstr += "\n\n\n\n"

        print(sstr)

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
