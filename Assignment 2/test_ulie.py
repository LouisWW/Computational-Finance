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

T, S0, K, r, sigma, steps = 1, 100, 99, 0.06, 0.2, 365
epsilons = [(x + 1) * 0.01 for x in range(50)]
# set_seed = "random"
set_seed = "fixed"
seed_nr = 10
iterations = [1000, 10000, 100000, 1000000, 10000000]
show_plot, save_plot, save_output = False, True, True
diff_iter = len(iterations)
eps_select = [0, 1, 4, 49]

start = time.time()
results = helper.diff_iter_bump_and_revalue(
    T=T,
    S0=S0,
    K=K,
    r=r,
    sigma=sigma,
    steps=steps,
    epsilons=epsilons, 
    set_seed=set_seed,
    seed_nr=seed_nr,
    iterations=iterations,
    option_type="regular",
    contract="put",
    show_plot=show_plot,
    save_plot=save_plot,
    save_output=save_output
)
print("Bump and Revalue EU Put Option ran in {} minutes"
        .format(round((time.time() - start) / 60, 2)))

# deltas, bs_deltas, errors, deviations = results

start2 = time.time()
results = helper.diff_iter_bump_and_revalue(
    T=T,
    S0=S0,
    K=K,
    r=r,
    sigma=sigma,
    steps=steps,
    epsilons=epsilons,
    set_seed=set_seed,
    seed_nr=seed_nr,
    iterations=iterations,
    option_type="digital",
    contract="call",
    show_plot=show_plot,
    save_plot=save_plot,
    save_output=save_output
)

print("Bump and Revalue EU Digital Call Option ran in {} minutes"
      .format(round((time.time() - start2) / 60, 2)))

# deltas, bs_deltas, errors, deviations = results

start3 = time.time()
results = helper.LR_method(
    T=T,
    S0=S0,
    K=K,
    r=r,
    sigma=sigma,
    steps=steps,
    set_seed=set_seed,
    seed_nr=seed_nr,
    reps=iterations,
    show_plot=show_plot,
    save_plot=save_plot,
    save_output=save_output
)

print("LR method EU Digital Call Option ran in {} minutes"
      .format(round((time.time() - start3) / 60, 2)))

# deltas, bs_deltas, errors, deviations = results

print("Entire script ran in {} minutes"
      .format(round((time.time() - start) / 60, 2)))


'''
Variance Reduction:
- control variates technique on Asian option based on arithmetic average
- study performance for different number of paths/strike/number of time points, etc.
'''




