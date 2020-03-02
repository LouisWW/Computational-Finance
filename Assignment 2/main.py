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

# helper.plot_wiener_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=True)
'''
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

'''
#helper.milstein_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)

#helper.antithetic_monte_carlo_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)

'''
Estimation of Sensitivities in MC:
-Compute bump-and-reveal mehtod
-Determine small delta
- use different/same seed for bumped/unbumped estimate of the value
- and point 2 use sophisticated method discussed in the lecture
'''

set_seed = []
reps = [10000, 100000, 1000000, 10000000]
# set_seed = [10] * len(reps)

results = helper.LR_method(
    T=1,
    S0=100,
    K=99,
    r=0.06,
    sigma=0.2,
    steps=365,
    set_seed=set_seed,
    reps=reps
)
deltas, bs_delta, errors = results
print("Monte Carlo Deltas:")
print(deltas.round(3))
print("=================================================")
print("Black Scholse Deltas:")
print(round(bs_delta, 3))
print("=================================================")
print("Relative Errors:")
print(errors.round(3))
print("=================================================")

epsilons = [0.01, 0.02, 0.5]
set_seed = []

# epsilons = [0.01 * (x + 1) for x in range(5)]
# set_seed = [10] * len(epsilons)


# results = helper.diff_iter_bump_and_revalue(
#     T=1,
#     S0=100,
#     K=99,
#     r=0.06,
#     sigma=0.2,
#     steps=365,
#     epsilons=epsilons,
#     set_seed=set_seed,
#     iterations=[10000, 100000, 1000000, 10000000],
#     full_output=False,
#     option_type="regular",
#     contract="put",
#     save_output=False
# )

results = helper.diff_iter_bump_and_revalue(
    T=1,
    S0=100,
    K=99,
    r=0.06,
    sigma=0.2,
    steps=365,
    epsilons=epsilons,
    set_seed=set_seed,
    iterations=[10000, 100000, 1000000, 10000000],
    full_output=False,
    option_type="digital",
    contract="call",
    save_output=False
)

deltas, bs_deltas, errors = results
print("Monte Carlo Deltas:")
print(deltas.round(3))
print("=================================================")
print("Black Scholse Deltas:")
print(bs_deltas.round(3))
print("=================================================")
print("Relative Errors:")
print(errors.round(3))
print("=================================================")


'''
Variance Reduction:
- control variates technique on Asian option based on arithmetic average
- study performance for different number of paths/strike/number of time points, etc.
'''
