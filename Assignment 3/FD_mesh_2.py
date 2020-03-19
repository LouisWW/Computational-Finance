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
import scipy.linalg as linalg
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

    def __init__(self, s_min, s_max, ds, t_max, dt, S0, K, r, sigma,option='call',fm_type='forward'):
        assert s_max !=0, "s_max needs to be greater than 0 !!"
        assert fm_type in ['forward','crank-nicolson'], "The finite method needs to be either 'forward' or 'crank-nicolson' "

        self.sigma = sigma
        self.K = K
        self.r = r
        self.S0 = S0

        self.s_max = s_max
        self.s_min = s_min
        self.ds = ds
        self.stock_prices = np.arange(s_min,s_max+ds,ds)  # +ds to include s_max

        self.t_max = t_max
        self.dt = dt
        self.t = np.arange(0,t_max+dt,dt)

        self.option = option
        self.fm_type = fm_type
        self.n_steps_s = len(np.arange(s_min, s_max+ds, ds))  # +ds to include s_max
        self.n_steps_t = len(np.arange(0,(t_max +dt), dt))

        # To make the grid from 0 to T_max/S_max
        self.grid = np.zeros((self.n_steps_s, self.n_steps_t))
        self.delta = np.zeros((self.n_steps_s, self.n_steps_t))



    def init_mesh(self):
        '''
        Set the boundary of the grid
        :return: self.grid
        '''

        if self.option == 'call':
            self.grid[:, -1] = np.array([max(0, i - self.K) for i in self.stock_prices])
            self.grid[0, :] = 0
            self.grid[-1, :] = np.array([(self.s_max-self.K)*math.exp(-self.r * t) for t in self.t[::-1]])

        elif self.option == 'put':
            self.grid[:, -1] = np.array([max(0, self.K-i) for i in self.stock_prices])
            self.grid[-1, :] = 0
            self.grid[0, :] = np.array([(self.K - self.s_max) * math.exp(-self.r * t) for t in self.t[::-1]])


    def coefficient(self):
        '''
        Calculate the coefficient derived in Hull, J. C. (2003). Options futures and other derivatives.
        Pearson Education India page 435
        :return: self.A, offset_coeeficient
        '''

        if self.fm_type == 'forward':

            v=np.arange(1,len(self.stock_prices)-1)
            self.A = np.zeros((self.n_steps_s-2, self.n_steps_s-2))

            self.alpha = -0.5 * self.r * v * self.dt+ 0.5 * self.sigma**2 * self.dt * v**2
            self.beta = 1 - self.dt * (self.sigma**2 * v**2 + self.r)
            self.gamma = .5 * self.r * v * self.dt + .5 * self.sigma**2 * v**2 * self.dt


            self.A += np.diag(self.beta, 0)
            self.A += np.diag(self.alpha[1:], -1)
            self.A += np.diag(self.gamma[0:-1], 1)

            anw = input("Do you want to print the Matrix ? y/n")
            if anw == 'y':
                print("Matrix A")
                self.print_matrix(self.A)


        elif self.fm_type == 'crank-nicolson':

            v = np.arange(1, len(self.stock_prices) - 1)
            self.A = np.zeros((self.n_steps_s-2, self.n_steps_s-2))
            self.B = np.zeros((self.n_steps_s-2, self.n_steps_s-2))

            self.alpha = (self.dt/4) * (self.sigma**2 * v**2 - self.r * v)
            self.beta = (-self.dt/2) * (self.sigma**2 * v**2 + self.r)
            self.gamma = (self.dt/4) * (self.sigma**2 * v**2 + self.r * v)

            self.A += np.diag(1+self.beta, 0)
            self.A += np.diag(self.alpha[1:], -1)
            self.A += np.diag(self.gamma[0:-1], 1)
            self.B += np.diag(1-self.beta, 0)
            self.B += np.diag(-self.alpha[1:], -1)
            self.B += np.diag(-self.gamma[0:-1], 1)

            # LU decomposition
            _,self.L,self.U = linalg.lu(self.B)

            anw = input("Do you want to print the Matrix ? y/n")
            if anw == 'y':
                print("Matrix A")
                self.print_matrix(self.A)
                print("\n\n\nMatrix B")
                self.print_matrix(self.B)


    def run(self):

        if self.option == 'call':
            cal_option_price=BlackScholes(self.t_max, self.S0, self.K, self.r, self.sigma).call_price()
        elif self.option == 'put':
            cal_option_price = BlackScholes(self.t_max, self.S0, self.K, self.r, self.sigma).put_price()



        # Setup
        self.init_mesh()
        self.coefficient()

        if self.fm_type == 'forward':
            for j in range(self.n_steps_t-1, 0, -1):
                self.grid[1:-1, j-1] = self.A.dot(self.grid[1:-1, j])


            comp_option_price=np.interp(self.S0,self.stock_prices,self.grid[:,0])
            print("\nThe analytical solution is {0:.3f} for a {1} option \n".format(cal_option_price, self.option))
            print("\nThe computed solution is {0:.3f} for a {1} option usign the {2} method \n".format( \
                comp_option_price, self.option, self.fm_type))

        elif self.fm_type == 'crank-nicolson':
            inner_grid=np.zeros(self.n_steps_s-2)
            for j in range(self.n_steps_t - 1, -1, -1):

                inner_grid[0]=self.alpha[1]*self.grid[0,j-1]+self.grid[0,j]
                inner_grid[-1]=self.gamma[-1]*(self.grid[-1,j-1]+self.grid[-1,j])

                mat_1=(np.dot(self.A,self.grid[1:-1,j])+inner_grid)
                mat_2=linalg.lstsq(self.L,mat_1)[0]
                mat_3=linalg.lstsq(self.U,mat_2)[0]

                self.grid[1:-1,j-1]= mat_3


            comp_option_price=np.interp(self.S0,self.stock_prices,self.grid[:,0])
            print("\nThe analytical solution is {0:.3f} for a {1} option \n".format(cal_option_price, self.option))
            print("\nThe computed solution is {0:.3f} for a {1} option usign the {2} method \n".format( \
                comp_option_price, self.option, self.fm_type))



    def print_matrix(self,matrix):

        str_matrix = "\n\n"
        # To make sure the coordinate (0,0) is at the bottom left
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                str_matrix += '%.3f' % matrix[i][j]
                str_matrix += "      "
            str_matrix += "\n"

        str_matrix += "\n\n\n\n"

        print(str_matrix)

    def __str__(self):

        self.str="Finite Difference Mesh for {} option \n\n".format(self.option)

        self.str += " S \ T      "
        for i in np.linspace(0,self.t_max,self.n_steps_t):
            self.str += '%.3f' % i + "      "
        self.str += "\n\n"

        # To make sure the coordinate (0,0) is at the bottom left
        for i in range(self.grid.shape[0]-1,-1,-1):
            if self.stock_prices[i]<100 and self.stock_prices[i]>=10:
                self.str += '%.3f' % self.stock_prices[i]
            elif self.stock_prices[i]<10:
                self.str += '%.4f' % self.stock_prices[i]
            else:
                self.str += '%.2f' % self.stock_prices[i]

            self.str += "      "

            for j in range(self.grid.shape[1]):

                if self.grid[i][j]<10:
                    self.str += '%.3f' % (self.grid[i][j])
                else:
                    self.str += '%.2f' % (self.grid[i][j])

                self.str += "      "
            self.str += "\n"

        self.str += "\n\n\n\n"

        return self.str
