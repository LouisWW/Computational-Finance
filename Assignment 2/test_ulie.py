#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import time
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

# epsilons = [0.01, 0.02, 0.5]
epsilons = [(x + 1) * 0.01 for x in range(50)]
# set_seed = []

# epsilons = [0.01 * (x + 1) for x in range(5)]
set_seed = [10] * len(epsilons)

start = time.time()
results = helper.diff_iter_bump_and_revalue(
    T=1,
    S0=100,
    K=99,
    r=0.06,
    sigma=0.2,
    steps=365,
    epsilons=epsilons, 
    set_seed=set_seed,
    iterations=[1000, 10000, 100000, 1000000, 10000000],
    full_output=False,
    option_type="regular",
    contract="put", 
    save_output=True
)
print("Bump and Revalue EU Put Option ran in {} minutes"
        .format(round((time.time() - start) / 60, 2)))


deltas, bs_deltas, errors, deviations = results
print("EU PUT OPTION (bump and revalue)")
print("==================================================")
print("Monte Carlo Deltas:")
print(deltas.round(3))
print("=================================================")
print("Black Scholse Deltas:")
print(bs_deltas.round(3))
print("=================================================")
print("Relative Errors:")
print(errors.round(3))
print("=================================================")
print("=================================================")
print("Standard Deviation Delta:")
print(deviations.round(3))
print("=================================================")

start2 = time.time()
results = helper.diff_iter_bump_and_revalue(
    T=1,
    S0=100,
    K=99,
    r=0.06,
    sigma=0.2,
    steps=365,
    epsilons=epsilons,
    set_seed=set_seed,
    iterations=[1000, 10000, 100000, 1000000, 10000000],
    full_output=False,
    option_type="digital",
    contract="call",
    save_output=True
)

print("Bump and Revalue EU Digital Call Option ran in {} minutes"
      .format(round((time.time() - start2) / 60, 2)))

deltas, bs_deltas, errors, deviations = results
print("DIGITAL EU CALL OPTION (bump and revalue")
print("==================================================")
print("Monte Carlo Deltas:")
print(deltas.round(3))
print("=================================================")
print("Black Scholse Deltas:")
print(bs_deltas.round(3))
print("=================================================")
print("Relative Errors:")
print(errors.round(3))
print("=================================================")
print("=================================================")
print("Standard Deviation Delta:")
print(deviations.round(3))
print("=================================================")

# set_seed = []
reps = [1000, 10000, 100000, 1000000, 10000000]
set_seed = [10] * len(reps)

start3 = time.time()
results = helper.LR_method(
    T=1,
    S0=100,
    K=99,
    r=0.06,
    sigma=0.2,
    steps=365,
    set_seed=set_seed,
    reps=reps,
    save_output=True
)

print("LR method EU Digital Call Option ran in {} minutes"
      .format(round((time.time() - start3) / 60, 2)))

deltas, bs_delta, errors, deviations = results
print("DIGITAL EU CALL OPTION (LR method)")
print("==================================================")
print("Monte Carlo Deltas:")
print(deltas.round(3))
print("=================================================")
print("Black Scholse Deltas:")
print(bs_deltas.round(3))
print("=================================================")
print("Relative Errors:")
print(errors.round(3))
print("=================================================")
print("Standard Deviation Delta:")
print(deviations.round(3))
print("=================================================")

print("Entire script ran in {} minutes"
      .format(round((time.time() - start) / 60, 2)))

'''
Variance Reduction:
- control variates technique on Asian option based on arithmetic average
- study performance for different number of paths/strike/number of time points, etc.
'''




