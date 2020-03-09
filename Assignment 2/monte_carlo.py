#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""


import numpy as np
import math



class monte_carlo:
    def __init__(self, steps, T, S0, sigma, r, K, market="EU", option_type="call"):
        self.steps = steps
        self.T = T
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.K = K
        self.dt = T / steps
        self.price = S0
        self.market = market.upper()
        self.option_type = option_type.lower()
        assert self.market in ["EU", "USA"], "Market not found. Choose EU or USA"
        assert self.option_type in ["call", "put"], "Non-existing option type."

    def wiener_method(self):
        """
        """
        price = self.price
        self.wiener_price_path = np.zeros(self.steps)
        for i in range(self.steps):
            self.wiener_price_path[i] = price
            ds = self.r * price * self.dt + self.sigma * \
                price * np.random.normal(0, 1) * np.sqrt(self.dt)
            price += ds

    def euler_integration_method(self,generate_path=False):

        self.euler_integration= self.S0 * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma *
                                                 np.sqrt(self.T)*np.random.normal(0, 1))

        if generate_path:
            price = self.price
            self.euler_price_path = np.zeros(self.steps)
            for i in range(self.steps):
                self.euler_price_path[i] = price
                ds = price * math.exp((self.r-0.5*self.sigma**2)*self.dt+
                                    self.sigma*np.random.normal(0, 1) * math.sqrt(self.dt))
                price = ds

            return self.euler_integration,self.euler_price_path

        return self.euler_integration

    def euler_method_vectorized(self, random_numbers):

        self.euler_vectorized = self.S0 * np.exp((self.r - 0.5 * self.sigma**2)
                                            * self.T + self.sigma * random_numbers)

        return self.euler_vectorized

    def milstein_method(self):
        """
        """
        price = self.price
        self.milstein_price_path = np.zeros(self.steps)
        for i in range(self.steps):
            self.milstein_price_path[i] = price
            ds = (1 + (self.r-.5*self.sigma**2)*self.dt + self.sigma*np.random.normal(0, 1)*np.sqrt(self.T) +
                  0.5*self.sigma**2*np.random.normal(0, 1)**2*self.dt)
            price = price * ds



    def antithetic_wiener_method(self):
        """
        """
        n_paths=1000
        paths_list=[]
        price = self.price
        anti_price= self.price
        for k in range(int(n_paths/2)):
            wiener_price_path = np.zeros(self.steps)
            anti_wiener_price_path = np.zeros(self.steps)

            for i in range(self.steps):
                epsilon=np.random.normal(0, 1)
                wiener_price_path[i] = price
                anti_wiener_price_path[i] = anti_price
                ds = self.r * price * self.dt + self.sigma * price * epsilon * np.sqrt(self.dt)
                anti_ds = self.r * anti_price * self.dt + self.sigma * price * -epsilon * np.sqrt(self.dt)

                price += ds
                anti_price += anti_ds

            paths_list.append(wiener_price_path)
            paths_list.append(anti_wiener_price_path)

        return paths_list
