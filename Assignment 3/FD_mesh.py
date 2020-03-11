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

    def __init__(self, s_min, s_max, ds, t_max, dt, r=0.06, sigma=0.2):
        assert s_max !=0 , "s_max needs to be greater than 0 !!"

        self.sigma =sigma
        self.r = r
        self.s_max = s_max
        self.s_min = s_min
        self.ds = ds
        self.t_max = t_max
        self.dt = dt
        self.n_steps_s = len(np.arange(self.s_min, self.s_max, self.ds))
        self.n_steps_t = int((self.t_max + 1) / dt)

        # To make the grid from 0 to T_max/S_max
        self.grid = np.zeros((self.n_steps_s, self.n_steps_t))
        self.delta = np.zeros((self.n_steps_s, self.n_steps_t))
        self.A = np.zeros((self.n_steps_s, self.n_steps_s))
        self.K = np.zeros(self.n_steps_s)

    def initialize_FTCS(self):
        """Setup, boundarie conditions are set to aviod 100 variables"""
        first = [{'value': -1, 'offset': 1}, {'value': 1, 'offset': -1}]
        second = [{'value': 2, 'offset': 0}, {'value': 1, 'offset': -1}, {'value': 1, 'offset': 1}]

        # Extra term for boundarie in first order matrix approx
        extra1 = np.zeros(self.n_steps_s)
        extra1[-1] = np.exp(self.t_max)

        # Extra term for boundarie in second order matrix approx
        extra2 = np.zeros(self.n_steps_s)
        extra2[-1] = np.exp(self.t_max) * (2/self.dt)

        # second and first order approximations in  matrix form with boundaries given
        part1 = self.tri_diag_matrix_func(0, 0, 0, 0, first, printing=True)  * (1 / (self.ds * 2)) + extra1
        part2 = self.tri_diag_matrix_func(0, 0, 2, -2, second, printing=True)  * (1 / (self.ds ** 2)) + extra2

        # K- term, interest and the A matrix price movement
        self.K += np.exp(self.r)
        self.A = (self.r - ((self.sigma ** 2) / 2)) * part1 + ((self.sigma ** 2) / 2) * part2

        # First layer in the grid
        self.grid[:, 0] = np.arange(self.s_min, self.s_max, self.ds)[::-1]

        # Show initial grid
        print("Initial")
        print(pd.DataFrame(self.grid))
        print(list(self.K))


    def forward_approximation(self, j):
        '''Forward approximation using the matrixes'''
        # Get old values and calculate new ones
        V_n = self.grid[:, j - 1]
        deltas = np.dot(V_n, self.A)
        V = V_n + self.dt * (deltas + self.K)

        # Save new values and delta
        self.grid[:, j] = V
        self.delta[:, j] = deltas

    def run(self):
        # Setup
        self.initialize_FTCS()

        # Loop, forward
        for j in range(1, self.grid.shape[1]):
            self.forward_approximation(j)

        # show results
        print("Final")
        print(pd.DataFrame(self.grid))


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
        tri_diag_matrix = np.zeros((self.n_steps_s, self.n_steps_s))

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
            print("matrix A")
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
