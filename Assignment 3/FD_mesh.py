#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import math


class BlackScholes:
    def __init__(self, T, S0, K, r, sigma, steps=1):
        self.T = T
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.dt = T / steps
        self.price = S0
        self.price_path = np.zeros(steps)

        self.delta_list = None
        self.x_hedge = None

    def call_price(self, t=0):
        """
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.sigma * np.sqrt(self.T - t)

        call = (self.S0 * st.norm.cdf(d1, 0.0, 1.0) - self.K *
                np.exp(-self.r * self.T) * st.norm.cdf(d2, 0.0, 1.0))

        return call

    def put_price(self, t=0):
        """
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.sigma * np.sqrt(self.T - t)

        put = ((self.K * np.exp(-self.r * self.T)
                * st.norm.cdf(-d2, 0.0, 1.0)) - self.S0 *
               st.norm.cdf(-d1, 0.0, 1.0))

        return put

    def asian_call_price(self, t=0) :
        """
        """
        N = self.steps
        sigma = self.sigma * np.sqrt(((N + 1) * (2 * N + 1)) / (6 * N ** 2))
        b = ((N + 1) / (2 * N)) * (self.r - 0.5 * (sigma ** 2))

        d1 = ((np.log(self.K / self.S0) + (b + 0.5 * self.sigma ** 2) * (self.T - t)) /
              (sigma * np.sqrt(self.T - t)))

        d2 = d1 - sigma * np.sqrt(self.T - t)

        call = (self.S0 * np.exp((b - self.r) * self.T) * st.norm.cdf(d1, 0.0, 1.0) - self.K *
                np.exp(-self.r * self.T) * st.norm.cdf(d2, 0.0, 1.0))

        return call

    def asian_put_price(self, t=0) :
        """
        """
        N = self.steps
        sigma = self.sigma * np.sqrt(((N + 1) * (2 * N + 1)) / (6 * N ** 2))
        b = ((N + 1) / (2 * N)) * (self.r - 0.5 * (sigma ** 2))

        d1 = ((np.log(self.K / self.S0) + (b + 0.5 * sigma ** 2) * (self.T - t)) /
              (sigma * np.sqrt(self.T - t)))

        d2 = d1 - (sigma * np.sqrt(self.T - t))

        put = self.K * np.exp(-self.r * self.T) * st.norm.cdf(-d2, 0.0, 1.0) - (
                    self.S0 * np.exp((b - self.r) * self.T) * st.norm.cdf(-d1, 0.0, 1.0))
        return put

    def create_price_path(self):
        """
        """
        for i in range(self.steps):
            self.price_path[i] = self.price
            dS = self.r * self.price * self.dt + self.sigma * \
                self.price * np.random.normal(0, 1) * np.sqrt(self.dt)

            self.price += dS


class FdMesh:

    def __init__(self, s_min, s_max, ds, t_max, dt, r=0.06, sigma=0.2, strike=100):
        assert s_max !=0 , "s_max needs to be greater than 0 !!"

        self.strike = strike
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

    def initialize_FTCS(self, option='put'):
        """Setup, boundarie conditions are set to aviod 100 variables"""
        first = [{'value': -1, 'offset': 1}, {'value': 1, 'offset': -1}]
        second = [{'value': 2, 'offset': 0}, {'value': 1, 'offset': -1}, {'value': 1, 'offset': 1}]

        # Extra term for boundary in first order matrix approx
        extra1 = np.zeros(self.n_steps_s)
        # extra1[-1] = np.exp(self.t_max)

        # Extra term for boundary in second order matrix approx
        extra2 = np.zeros(self.n_steps_s)
        # extra2[-1] = np.exp(self.t_max) * (2/self.dt)

        # second and first order approximations in  matrix form with boundaries given
        part1 = self.tri_diag_matrix_func(0, 0, 0, 0, first, printing=True) * (1 / (self.ds * 2)) + extra1
        part2 = self.tri_diag_matrix_func(0, 0, 2, -2, second, printing=True) * (1 / (self.ds ** 2)) + extra2

        # K- term, interest and the A matrix price movement
        self.K += self.r
        self.A = (self.r - ((self.sigma ** 2) / 2)) * part1 + ((self.sigma ** 2) / 2) * part2

        # First layer in the grid
        stock_prices = np.arange(self.s_min, self.s_max, self.ds)[::-1]
        if option == 'put':
            first = np.array([max(0, i - self.strike) for i in stock_prices])

        self.grid[:, 0] = first

        # Show initial grid
        print("Initial")
        print(pd.DataFrame(self.grid))
        print(list(self.K))


    def forward_approximation(self, j):
        '''Forward approximation using the matrixes'''
        # Get old values and calculate new ones
        V_n = self.grid[:, j - 1]
        deltas = np.dot(V_n, self.A)
        V = V_n + self.dt * (deltas - self.K * V_n)

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

        # add offset with correct values over the given diagonal
        for set in offsets:
            array_len = (tri_diag_matrix.shape[1] - abs(set['offset']))
            tri_diag_matrix += np.diag([set['value']] * array_len, set['offset'])

        # the size of the matrix depends only on the number of discrete stock prices
        tri_diag_matrix[0][0] = k1
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
        for i in range(self.grid.shape[0]-1,-1,-1):
            for j in range(self.grid.shape[1]):

                self.str += str(self.grid[i][j])
                self.str += "  "
            self.str += "\n"

        self.str += "\n\n\n\n"

        return self.str
