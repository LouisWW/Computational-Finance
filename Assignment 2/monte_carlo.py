#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""


import numpy as np

class monte_carlo:
    def __init__(self, N, T, S0, sigma, r, K,market="EU", option_type="call"):
        self.N = N
        self.T = T
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.K = K
        self.market = market.upper()
        self.option_type = option_type.lower()
        self.array_out = array_out
        assert self.market in ["EU", "USA"], "Market not found. Choose EU or USA"
        assert self.option_type in ["call", "put"], "Non-existing option type."


