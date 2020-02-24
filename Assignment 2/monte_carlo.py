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
        self.price_path = np.zeros(steps)
        self.market = market.upper()
        self.option_type = option_type.lower()
        assert self.market in ["EU", "USA"], "Market not found. Choose EU or USA"
        assert self.option_type in ["call", "put"], "Non-existing option type."

    def wiener_method(self):
            """
            """
            for i in range(self.steps):
                self.price_path[i] = self.price
                dS = self.r * self.price * self.dt + self.sigma * \
                     self.price * np.random.normal(0, 1) * np.sqrt(self.dt)

                self.price += dS

    def euler_method(self):

        self.S = self.price

        for i in range(self.steps):

            self.price_path[i] = self.S
            dS = self.S*(self.r*self.dt+self.sigma*np.random.normal(0, 1) * np.sqrt(self.dt))

            self.S += dS


    def euler_integration(self):

       self.euler_integration = self.S0 * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma *
                                                 np.sqrt(self.T)*np.random.normal(0, 1) * np.sqrt(self.dt))






