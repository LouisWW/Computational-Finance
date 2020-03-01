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
    T, S0, K, r, sigma, steps, epsilons, set_seed=None, reps=100, 
    full_output=False, save_plot=False, show_plots=False, option_type="put"
):
    """
    """
    
    # Init amount of bumps (epsilons) and setup tracking variables
    # for the deltas generated in the monte carlo simulations
    diff_eps = len(epsilons)
    deltas = np.zeros(diff_eps)
    bs_deltas = np.zeros(diff_eps)

    # Start MC simulation for each bump
    for i, eps in enumerate(epsilons):

        # Determine "bumped" price
        S0_eps = S0 + eps

        # Set seed if given by user
        if set_seed is not None:
            np.random.seed(set_seed)

        # Generate random numbers of given amount of repititions
        numbers = np.random.normal(size=reps)

        # Create bump and revalue Monte Carlo (MC) objects
        mc_revalue = monte_carlo(steps, T, S0, sigma, r, K)
        mc_bump = monte_carlo(steps, T, S0_eps, sigma, r, K)

        # Euler method
        S_rev = mc_revalue.euler_method_vectorized(numbers)
        S_bump = mc_bump.euler_method_vectorized(numbers)

        # Determine prices and delta hedging depending on option type
        prices_revalue, prices_bump = 0, 0
        if option_type == "digital":
            prices_revalue = digital_call_price(K, S_rev, r, T)
            prices_bump = digital_call_price(K, S_bump, r, T)
            d2 = (np.log(S0_eps / K) + (r - 0.5 * sigma ** 2)
                  * T) / (sigma * np.sqrt(T))
            num = math.exp(-r * T) * stats.norm.pdf(d2, 0.0, 1.0)
            den = sigma * S0_eps * math.sqrt(T)
            bs_deltas[i] = num / den
        else:
            prices_revalue = put_price(K, S_rev, r, T)
            prices_bump = put_price(K, S_bump, r, T)
            d1 = (np.log(S0_eps / K) + (r + 0.5 * sigma ** 2)
                  * T) / (sigma * np.sqrt(T))
            bs_deltas[i] = -stats.norm.cdf(-d1, 0.0, 1.0)

        mean_revalue = prices_revalue.mean()
        mean_bump = prices_bump.mean()

        # Determine MC delta and theoretical delta ()
        deltas[i] = (mean_bump - mean_revalue) / eps

    # Determine relative errors
    errors = np.abs(1 - deltas / bs_deltas)

    return deltas, bs_deltas, errors

def put_price(K, S, r, T):
    """
    """
    return math.exp(-r * T) * np.where(K - S > 0, K - S, 0)

def digital_call_price(K, S, r, T):
    """
    """
    return math.exp(-r * T) * np.where(S - K > 0, 1, 0)

def bump_and_revalue(
        T, S0, K, r, sigma, steps, 
        epsilons, set_seed=None, reps=100, full_output=False, save_plot=False, show_plots=False
    ):
    """
    """

    # Init amount of bumps (epsilons) and setup tracking variables
    # for the prices and deltas generated in the monte carlo simulations
    diff_eps = len(epsilons)
    prices_revalue = np.zeros((reps, diff_eps))
    prices_bump = np.zeros((reps, diff_eps))
    deltas = np.zeros(diff_eps)
    bs_deltas = np.zeros(diff_eps)

    # Start simulation for each bump (eps)
    for i, eps in enumerate(epsilons):

        # "bumped" stock price
        S_eps = S0 + eps

        for j in range(reps):

            # Create bump and revalue Monte Carlo (MC) objects
            mc_revalue = monte_carlo(steps, T, S0, sigma, r, K)
            mc_bump = monte_carlo(steps, T, S_eps, sigma, r, K)

            # Euler integration MC rev, save discounted payoff at maturity
            mc_revalue.euler_integration_method()
            payoff_revalue = max([K - mc_revalue.euler_integration, 0])
            prices_revalue[j, i] = math.exp(-r * T) * payoff_revalue

            # Euler integration MC bump, save discounted payoff at maturity
            mc_bump.euler_integration_method()
            payoff_bump = max([K - mc_bump.euler_integration, 0])
            prices_bump[j, i] = math.exp(-r * T) * payoff_bump
 
        # Takes mean of prices bump and revalue and determines the delta 
        # for a given bump
        mean_price_revalue = prices_revalue[:, i].mean()
        mean_price_bump = prices_bump[:, i].mean()
        print("=============================================")
        print(f"REVALUE WITH EPS = {eps}")
        print(round(mean_price_revalue, 3))
        print("=============================================")
        print(f"BUMP WITH EPS = {eps}")
        print(round(mean_price_bump, 3))
        print("=============================================")
        deltas[i] = (mean_price_bump - mean_price_revalue) / eps

        # Determine theoretical delta for given bump
        d1 = (np.log((S0 + eps) / K) + (r + 0.5 * sigma ** 2)
              * T) / (sigma * np.sqrt(T))
        bs_deltas[i] = -stats.norm.cdf(-d1, 0.0, 1.0)

    # Makes histograms of the prices generated by MC process
    if show_plots:
        prices = (prices_revalue, prices_bump)
        for i in range(len(prices)):
            for j, eps in enumerate(epsilons):
                fig = plt.figure()
                plt.hist(prices[i][:, j], bins=10)
                plt.title(f"Epsilon = {eps}")
                plt.show()
                plt.close()


    if full_output:
        return prices_revalue, prices_bump, deltas, bs_deltas

    return deltas, bs_deltas

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

