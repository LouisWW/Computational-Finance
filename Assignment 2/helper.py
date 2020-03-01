#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import math
from monte_carlo import monte_carlo
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import tqdm
from collections import defaultdict
import multiprocessing
from Binomial_tree import BinTreeOption, BlackScholes
import tqdm
import pickle




def plot_wiener_process(T, S0, K, r, sigma, steps,save_plot=False):
    """
    :param T:  Period
    :param S0: Stock price at spot time
    :param K:  Strike price
    :param r:  interest rate
    :param sigma: volatility
    :param steps: number of steps
    :param save_plot:  to save the plot
    :return:  returns a plot of a simulated stock movement
    """

    mc=monte_carlo(steps, T, S0, sigma, r, K)

    mc.wiener_method()


    plt.figure()
    np.linspace(1,mc.T*365,mc.steps)#to ensure the x-axis is in respective to the total time T
    plt.plot(np.linspace(1,mc.T*365,mc.steps),mc.wiener_price_path)
    plt.xlabel("Days",fontsize=12,fontweight='bold')
    plt.ylabel("Stock price",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title("Stock price simulated based on the Wiener process",fontsize=14,fontweight='bold')
    if save_plot:
        plt.savefig("figures/"+"wiener_process",dpi=300)
    plt.show()
    plt.close()


def worker_pay_off_euler_direct(object):
    np.random.seed()
    object.euler_integration_method()
    pay_off_array = np.max([(object.K - object.euler_integration), 0])

    return pay_off_array

def worker_pay_off_euler_sim(object):
    np.random.seed()
    object.euler_integration_method(generate_path=True)
    pay_off_array = np.max([(object.K - object.euler_price_path[-1]), 0])

    return pay_off_array

def diff_monte_carlo_process(T, S0, K, r, sigma, steps,increments,max_repetition,save_plot=False):
    """
    :param T:  Period
    :param S0: Stock price at spot time
    :param K:  Strike price
    :param r:  interest rate
    :param sigma: volatility
    :param steps: number of steps
    :param save_plot:  to save the plot
    :return:  returns a plot of a simulated stock movement
    """

    different_mc_rep = np.linspace(10,max_repetition,increments,dtype=int)

    # mc_pricing will be a dict a list containing  tuples of (pricing and standard error)
    mc_pricing = defaultdict(list)

    for repetition in tqdm.tqdm(different_mc_rep):

        mc_list = [monte_carlo(steps, T, S0, sigma, r, K) for i in range(repetition)]
        num_core = 3
        pool = multiprocessing.Pool(num_core)
        pay_off_list = pool.map(worker_pay_off_euler_direct, ((mc) for mc in mc_list))
        pool.close()
        pool.join()

        mean_pay_off = np.mean([pay_off for pay_off in pay_off_list])
        std_pay_off = np.std([pay_off for pay_off in pay_off_list])/np.sqrt(repetition)
        mc_pricing['euler_integration'].append((np.exp(-r*T)*mean_pay_off ,std_pay_off))

    bs = BlackScholes(T, S0, K, r, sigma)
    bs_solution=np.ones(increments)*bs.put_price()

    fig, axs = plt.subplots(2)
    axs[0].plot(different_mc_rep, [i[0] for i in mc_pricing['euler_integration']], color='gray', label='Monte Carlo')
    axs[0].plot(different_mc_rep, bs_solution, 'r', label='Black Scholes')
    axs[0].legend()
    axs[0].set_ylabel("Option Price", fontsize=14)
    axs[0].tick_params(labelsize='15')

    axs[1].plot(different_mc_rep, [i[1] for i in mc_pricing['euler_integration']], label='Standard error')
    axs[1].set_xlabel("Monte Carlo repetition", fontsize=14)
    axs[1].legend()
    axs[1].set_ylabel("Standard error", fontsize=14)
    axs[1].tick_params(labelsize='15')

    if save_plot:
        plt.savefig("figures/" + "mc_euler_integration_diff_MC", dpi=300)
    plt.show()
    plt.close()



def diff_K_monte_carlo_process(T,different_k , S0, r, sigma, steps, repetition, save_plot=False):
    """
    :param T:  Period
    :param S0: Stock price at spot time
    :param K:  Strike price
    :param r:  interest rate
    :param sigma: volatility
    :param steps: number of steps
    :param save_plot:  to save the plot
    :return:  returns a plot of a simulated stock movement
    """

    # mc_pricing will be a dict of a list containing  tuples of (pricing and standard error)
    mc_pricing = defaultdict(list)

    for diff_strike_price in tqdm.tqdm(different_k):

        mc_list = [monte_carlo(steps, T, S0, sigma, r, diff_strike_price) for i in range(repetition)]
        num_core = 3
        pool = multiprocessing.Pool(num_core)
        pay_off_list = pool.map(worker_pay_off_euler_direct, ((mc) for mc in mc_list))
        pool.close()
        pool.join()

        mean_pay_off = np.mean([pay_off for pay_off in pay_off_list])
        std_pay_off = np.std([pay_off for pay_off in pay_off_list])/np.sqrt(repetition)
        mc_pricing['euler_integration'].append((np.exp(-r*T)*mean_pay_off,std_pay_off))

    bs_list= []
    for k in different_k:
        bs = BlackScholes(T, S0, k, r, sigma)
        bs_list.append(bs.put_price())

    fig, axs = plt.subplots(2)
    axs[0].plot(different_k,[i[0] for i in mc_pricing['euler_integration']],linestyle='--',linewidth=3,
                color='gray', label='Monte Carlo')
    axs[0].plot(different_k, bs_list, 'r', label='Black Scholes')
    axs[0].legend()
    axs[0].set_ylabel("Option Price",fontsize=14)
    axs[0].tick_params(labelsize='15')

    axs[1].plot(different_k,[i[1] for i in mc_pricing['euler_integration']],label='Standard error')
    axs[1].set_xlabel("Strike price K", fontsize=14)
    axs[1].legend()
    axs[1].set_ylabel("Standard error", fontsize=14)
    axs[1].tick_params(labelsize='15')
    axs[1].ticklabel_format(axis="y", style="sci",scilimits=(0,0))


    if save_plot:
        plt.savefig("figures/" + "mc_euler_integration_diff_K", dpi=300)
    plt.show()
    plt.close()

def diff_sigma_monte_carlo_process(T,K , S0, r, different_sigma, steps, repetition, save_plot=False):
    """
    :param T:  Period
    :param S0: Stock price at spot time
    :param K:  Strike price
    :param r:  interest rate
    :param sigma: volatility
    :param steps: number of steps
    :param save_plot:  to save the plot
    :return:  returns a plot of a simulated stock movement
    """

    # mc_pricing will be a dict of a list containing  tuples of (pricing and standard error)
    mc_pricing = defaultdict(list)

    for sigma in tqdm.tqdm(different_sigma):

        mc_list = [monte_carlo(steps, T, S0, sigma, r, K) for i in range(repetition)]
        num_core = 3
        pool = multiprocessing.Pool(num_core)
        pay_off_list = pool.map(worker_pay_off_euler_direct, ((mc) for mc in mc_list))
        pool.close()
        pool.join()

        mean_pay_off = np.mean([pay_off for pay_off in pay_off_list])
        std_pay_off = np.std([pay_off for pay_off in pay_off_list])/np.sqrt(repetition)
        mc_pricing['euler_integration'].append((np.exp(-r*T)*mean_pay_off,std_pay_off))

    bs_list = []
    for s in different_sigma:
        bs = BlackScholes(T, S0, K, r, s)
        bs_list.append(bs.put_price())

    fig, axs = plt.subplots(2)
    axs[0].plot(different_sigma,[i[0] for i in mc_pricing['euler_integration']],linestyle='--',linewidth=3,
                color='gray', label='Monte Carlo')
    axs[0].plot(different_sigma, bs_list, 'r', label='Black Scholes')
    axs[0].legend()
    axs[0].set_ylabel("Option Price",fontsize=14)
    axs[0].tick_params(labelsize='15')

    axs[1].plot(different_sigma,[i[1] for i in mc_pricing['euler_integration']],label='Standard error')
    axs[1].set_xlabel("Volatility", fontsize=14)
    axs[1].legend()
    axs[1].set_ylabel("Standard error", fontsize=14)
    axs[1].tick_params(labelsize='15')
    axs[1].ticklabel_format(axis="y", style="sci",scilimits=(0,0))


    if save_plot:
        plt.savefig("figures/" + "mc_euler_integration_diff_sigma", dpi=300)
    plt.show()
    plt.close()



def milstein_process(T, S0, K, r, sigma, steps,save_plot=False):
    """
    :param T:  Period
    :param S0: Stock price at spot time
    :param K:  Strike price
    :param r:  interest rate
    :param sigma: volatility
    :param steps: number of steps
    :param save_plot:  to save the plot
    :return:  returns a plot of a simulated stock movement
    """

    mc = monte_carlo(steps, T, S0, sigma, r, K)

    price_path=mc.milstein_method()

    plt.figure()
    np.linspace(1, mc.T * 365, mc.steps)  # to ensure the x-axis is in respective to the total time T
    plt.plot(np.linspace(1, mc.T * 365, mc.steps), mc.milstein_price_path)
    plt.xlabel("Days", fontsize=12, fontweight='bold')
    plt.ylabel("Stock price", fontsize=12, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title("Milestein method", fontsize=14, fontweight='bold')
    if save_plot:
        plt.savefig("figures/" + "milestein method", dpi=300)
    plt.show()
    plt.close()



def antithetic_monte_carlo_process(T, S0, K, r, sigma, steps,save_plot=False):

    mc = monte_carlo(steps, T, S0, sigma, r, K)

    path_list=mc.antithetic_wiener_method()

    plt.figure()
    plt.plot(path_list[0])
    plt.plot(path_list[1])
    plt.xlabel("Days", fontsize=12, fontweight='bold')
    plt.ylabel("Stock price", fontsize=12, fontweight='bold')
    plt.title("Antithetic Monte Carlo", fontsize=14, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()
    plt.close()

def bump_revalue_vectorized(
    T=1, S0=100, K=99, r=0.06, sigma=0.02, steps=365, epsilons=[0.5], 
    set_seed=[], reps=100, full_output=False, option_type="put"
):
    """
    """
    
    # Init amount of bumps (epsilons) and storage (Black Scholes) deltas
    diff_eps = len(epsilons)
    deltas = np.zeros(diff_eps)
    bs_deltas = np.zeros(diff_eps)

    # Start MC simulation for each bump
    for i, eps in enumerate(epsilons):

        # Determine "bumped" price
        S0_eps = S0 + eps

        # Create bump and revalue Monte Carlo (MC) objects
        mc_revalue = monte_carlo(steps, T, S0, sigma, r, K)
        mc_bump = monte_carlo(steps, T, S0_eps, sigma, r, K)

        # Determine stock prices at maturity
        S_rev, S_bump = stock_prices_bump_revalue(
                            set_seed, reps, mc_revalue, mc_bump
                        )

        # Determine prices and delta hedging depending at spot time
        results = option_prices_spot(
            option_type, S_rev, S_bump, S0_eps, K, r, sigma, T, bs_deltas, i
        )
        prices_revalue, prices_bump, bs_deltas = results

        # Mean option prices bump and revalue
        mean_revalue = prices_revalue.mean()
        mean_bump = prices_bump.mean()

        # Determine MC delta
        deltas[i] = (mean_bump - mean_revalue) / eps

    # Determine relative errors
    errors = np.abs(1 - deltas / bs_deltas)

    # Checks if full output is required
    if full_output:
        return deltas, bs_deltas, errors, prices_revalue, prices_bump

    return deltas, bs_deltas, errors

def stock_prices_bump_revalue(set_seed, reps, mc_revalue, mc_bump):
    """
    """
    # Set seed (if given) and generate similar sequence for bump and revalue
    S_rev, S_bump = None, None
    if set_seed:
        np.random.seed(set_seed[i])
        numbers = np.random.normal(size=reps)

        # Euler method
        S_rev = mc_revalue.euler_method_vectorized(numbers)
        S_bump = mc_bump.euler_method_vectorized(numbers)

    # Otherwise generate a different sequence for bump and revalue
    else:
        numbers_rev = np.random.normal(size=reps)
        numbers_bump = np.random.normal(size=reps)

        # Euler method
        S_rev = mc_revalue.euler_method_vectorized(numbers_rev)
        S_bump = mc_bump.euler_method_vectorized(numbers_bump)

    return S_rev, S_bump

def option_prices_spot(
    option_type, S_rev, S_bump, S0_eps, K, r, sigma, T, bs_deltas, i
    ):
    """
    Determine prices and delta hedging at spot time depending on option type
    """
    prices_revalue, prices_bump, discount = 0, 0, math.exp(-r * T)

    # Digital option
    if option_type == "digital":

        # Determine option price
        prices_revalue = discount * np.where(S_rev - K > 0, 1, 0)
        prices_bump = discount * np.where(S_bump - K > 0, 1, 0)

        # Theoretical delta
        d2 = (np.log(S0_eps / K) + (r - 0.5 * sigma ** 2)
                * T) / (sigma * np.sqrt(T))
        num = discount * stats.norm.pdf(d2, 0.0, 1.0)
        den = sigma * S0_eps * math.sqrt(T)
        bs_deltas[i] = num / den

    # European put option
    else:

        # Determine option price
        prices_revalue = discount * np.where(K - S_rev > 0, K - S_rev, 0)
        prices_bump = discount * np.where(K - S_bump > 0, K - S_bump, 0)

        # Theoretical delta
        d1 = (np.log(S0_eps / K) + (r + 0.5 * sigma ** 2)
                * T) / (sigma * np.sqrt(T))
        bs_deltas[i] = -stats.norm.cdf(-d1, 0.0, 1.0)

    return prices_revalue, prices_bump, bs_deltas


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

