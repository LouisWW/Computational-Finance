#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import helper as helper
import numpy as np


'''
Basic Option Valuation :
-computing the discounted value of the average pay-off
- convergence studies by increasing the number of trials and compare to assignment 1
- vary the strike price and the volatility
- estimate the Standard error and accuracy
'''

# helper.plot_wiener_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)

helper.diff_monte_carlo_process(
    T=1,
    S0=100,
    K=99,
    r=0.06,
    sigma=0.2,
    steps=365,
    increments=50,
    max_repetition=10000,
    save_plot=True)


helper.diff_K_monte_carlo_process(
    T=1,
    different_k=np.linspace(80,130,dtype=int),
    S0=100,
    r=0.06,
    sigma=0.2,
    steps=365,
    repetition=10000,
    save_plot=True)



helper.diff_sigma_monte_carlo_process(
    T=1,
    K=99,
    S0=100,
    r=0.06,
    different_sigma=np.linspace(0.01,1),
    steps=365,
    repetition=10000,
    save_plot=True)


#helper.milstein_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)

#helper.antithetic_monte_carlo_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)

'''
Estimation of Sensitivities in MC:
-Compute bump-and-reveal mehtod
-Determine small delta
- use different/same seed for bumped/unbumped estimate of the value
- and point 2 use sophisticated method discussed in the lecture
'''

results = helper.bump_revalue_vectorized(
    T=1, 
    S0=100, 
    K=99, 
    r=0.06, 
    sigma=0.02, 
    steps=365, 
    epsilons=[0.5], 
    set_seed=[], 
    reps=10000, 
    full_output=False, 
    option_type="put"
)
deltas, bs_deltas, errors = results
print("Monte Carlo Deltas:")
print(deltas)
print("=================================================")
print("Black Scholse Deltas:")
print(bs_deltas)
print("=================================================")
print("Relative Errors:")
print(errors)
print("=================================================")


'''
Variance Reduction:
- control variates technique on Asian option based on arithmetic average
- study performance for different number of paths/strike/number of time points, etc.
'''
