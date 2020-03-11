#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import numpy as np
import pandas as pd


class FdMesh:

    def __init__(self, s_max, ds, t_max, dt):
        assert s_max !=0 , "s_max needs to be greater than 0 !!"
        self.s_max = s_max
        self.ds = ds
        self.t_max = t_max
        self.dt = dt

        # To make the grid from 0 to T_max/S_max
        self.grid = np.zeros((int((self.t_max + 1) / dt), int((self.s_max + 1) / ds)))

    def forward_approximation(self):
        pass

    def first_derivitive_space(self, i, j):
        """Forward approximation of first derivative with respect to space:"""
        return (self.grid[i][j + 1] - self.grid[i][j - 1]) / (2 * self.ds)

    def first_derivitive_time(self, i, j):
        """Forward approximation of first derivative with respect to space:"""
        return (self.grid[i + 1][j] - self.grid[i - 1][j]) / (2 * self.ds)

    def second_derivitive_space(self, i, j):
        """central approximation of second derivative with respect to space:"""
        return (self.grid[i][j + 1] - self.grid[i][j] + self.grid[i][j - 1]) / (self.ds ** 2)

    def tri_diag_matrix_func(self, k1, k2, k3, k4, offsets, printing=False):

        # initialize
        tri_diag_matrix = np.zeros((self.grid.shape[1], self.grid.shape[1]))

        # add offset with correct values over the given diaganol
        for set in offsets:
            array_len = (tri_diag_matrix.shape[1] - abs(set['offset']))
            tri_diag_matrix += np.diag([set['value']] * array_len, set['offset'])

        # the size of the matrix depends only on the number of discrete stock prices
        tri_diag_matrix[0][0] =k1
        tri_diag_matrix[0][1] = k2
        tri_diag_matrix[-1][-2] = k3
        tri_diag_matrix[-1][-1] = k4

        # For testing
        if printing:
            print(pd.DataFrame(tri_diag_matrix))

        return tri_diag_matrix

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
