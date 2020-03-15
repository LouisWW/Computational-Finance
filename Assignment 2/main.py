#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import helper as helper
import numpy as np
import argparse



parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Price an option based on the Monte Carlo method with variance reduction given the spot price S0, the strike price K, \
the volatility sigma and the interest rate. \n\n \
The different - func: \n '
'-wiener_process : Plots the wiener process \
-diff_Mc_samples : Computes MC with different number of samples using the default parameter \n \
-diff_K : Computes MC with different strike price using the default parameter \n \
-diff_sigma : Computes MC with different implied volatility using the default parameter \n \
-lr_method : Computes the likelihood ration for discounted payoffs of digital option \n \
-bump_and_revalue : Use bump and revalue method to determine the Delta \n ' )

parser.add_argument("-func",type = str, default='diff_Mc_samples', help='Defines which function to execute')
parser.add_argument('-T', type=int,default=1, help='Time to maturity in years (default : 1)')
parser.add_argument('-S', type=int,default=100, help='Stoke price at the moment (default : 100)')
parser.add_argument('-K', type=int,default=99, help='Strike price at the moment (default : 99)')
parser.add_argument('-steps', type=int,default=365, help='Number of updates of the stock market over the year')
parser.add_argument('-r', type=float,default=0.06, help='interest rate r (default : 0.06)')
parser.add_argument('-s', type=float, default=0.2, help='volatility s (default : 0.2)')
parser.add_argument('-option_type', type=str,default='put', help='option type call or put (default : call)')
parser.add_argument('-market', type=str,default='EU', help='option type EU or USA(default : EU)')
parser.add_argument('-save_plot',default=False, help='return the plots (default : False)')
parser.add_argument('-diff_samples',type=int, default=[100,1000,10000,100000,1000000], help='Different number of samples (default: default=[100,1000,10000,100000,1000000])')
parser.add_argument('-samples',type=int,default=10000,help='Number of samples (default: 10000)')
parser.add_argument('-different_k',type=int,default=np.linspace(80,130,dtype=int),help='Different strike price (default:80-130)')
parser.add_argument('-different_s',type=float,default=np.linspace(0.01,1),help='Different volatility from 0 to 1')
parser.add_argument('-epsilons',type=float,default= [0.01, 0.02, 0.5], help='set epsilon to bump the stock price for the bump and revalue method (default: [0.01, 0.02, 0.5])')
parser.add_argument('-set_seed',type=str,default='fixed',help='Set a seed (default : fixed or random')
parser=parser.parse_args()


if not parser.func in ['wiener_process','diff_Mc_samples','diff_K','diff_sigma','lr_method','bump_and_revalue'] :
    print("\n\n\n !!! You need to define a funciton that exists !!!  \n\n\n")
    raise AssertionError()


'''
Basic Option Valuation :
-computing the discounted value of the average pay-off
- convergence studies by increasing the number of trials and compare to assignment 1
- vary the strike price and the volatility
- estimate the Standard error and accuracy
'''
if parser.func == 'wiener_process':
    helper.plot_wiener_process(parser.T,parser.K, parser.S, parser.r, parser.s, parser.steps, parser.save_plot)

elif parser.func == 'diff_Mc_samples':
    helper.diff_monte_carlo_process(
        parser.T,
        parser.S,
        parser.K,
        parser.r,
        parser.s,
        parser.steps,
        parser.diff_samples,
        parser.save_plot)

elif parser.func == 'diff_K':
    helper.diff_K_monte_carlo_process(
        parser.T,
        parser.different_k,
        parser.S,
        parser.r,
        parser.s,
        parser.steps,
        parser.samples,
        parser.save_plot)


elif parser.func == 'diff_sigma':
    helper.diff_sigma_monte_carlo_process(
        parser.T,
        parser.K,
        parser.S,
        parser.r,
        parser.different_s,
        parser.steps,
        parser.repetition,
        parser.save_plot)

elif parser.func == 'lr_method':
    parser.set_seed = []
    if parser.set_seed:
        set_seed = [10] * len(reps)

    results = helper.LR_method(
        parser.T,
        parser.S,
        parser.K,
        parser.r,
        parser.s,
        parser.steps,
        parser.set_seed,
        parser.diff_samples,
        parser.option_type
    )
    deltas, bs_delta, errors, variances = results
    print("Monte Carlo Deltas:")
    print(deltas.round(3))
    print("=================================================")
    print("Black Scholse Deltas:")
    print(bs_delta)
    print("=================================================")
    print("Relative Errors:")
    print(errors.round(3))
    print("=================================================")

elif parser.func == 'bump_and_revalue':
    parser.set_seed = []
    if parser.set_seed:
        set_seed = [10] * len(reps)


    results = helper.diff_iter_bump_and_revalue(
         parser.T,
         parser.S,
         parser.K,
         parser.r,
         parser.s,
         parser.steps,
         parser.epsilons,
         parser.set_seed,
         parser.diff_samples,
         parser.option_type
    )


    deltas, bs_deltas, errors, variances = results
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
