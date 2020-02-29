#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import helper as helper



'''
Basic Option Valuation : 
-computing the discounted value of the average pay-off
- convergence studies by increasing the number of trials and compare to assignment 1
- vary the strike price and the volatility
- estimate the Standard error and accuracy
'''

#helper.plot_wiener_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)
#helper.monte_carlo_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)
#helper.diff_monte_carlo_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=True)
#helper.diff_K_monte_carlo_process(1, 100,99, 0.06, 0.2, steps=365,save_plot=True)

#helper.test(1, 100, 99, 0.06, 0.2, steps=365,save_plot=False)
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
    1, 100, 99, 0.06, 0.2, 365, [0.01, 0.02, 0.5], reps=100, full_output=True, save_plot=False, show_plots=False, set_seed=1
)
deltas, bs_deltas = results
print("Monte Carlo Deltas:")
print(deltas)
print("=================================================")
print("Black Scholse Deltas:")
print(bs_deltas)
print("=================================================")


# mean_deltas = helper.bump_and_revalue(
#     1, 100, 99, 0.06, 0.2, 365, [0.01, 0.02, 0.5], reps=100, full_output=True, save_plot=False, show_plots=False
#     )
# prices_revalue, prices_bump, deltas, bs_deltas = mean_deltas
# print(deltas)
# print("======================================================================")
# print(bs_deltas)
# print("=====================================================================")
# print("PRICES REVALUE")
# print(prices_revalue)
# print("=====================================================================")
# print("=====================================================================")
# print("PRICES BUMPS")
# print(prices_bump)
# print("=====================================================================")
# print("=====================================================================")
# print("DELTAS BUMPS")
# print(deltas)
# print("=====================================================================")
# print("=====================================================================")
# print("THEORETICAL DELTAS")
# print(bs_deltas)
# print("=====================================================================")


'''
Variance Reduction:
- control variates technique on Asian option based on arithmetic average
- study performance for different number of paths/strike/number of time points, etc.
'''




