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
from matplotlib import rc



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


def worker_pay_off_euler(object):
    np.random.seed()
    object.euler_integration_method()
    pay_off_array_sim = np.max([(object.K - object.euler_price_path[-1]), 0])
    pay_off_array_direct = np.max([(object.K - object.euler_integration), 0])

    return pay_off_array_sim,pay_off_array_direct



def diff_monte_carlo_process(T, S0, K, r, sigma, steps,save_plot=False):
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


    increments = 10
    max_repetition = 10000
    different_mc_rep = np.linspace(10,max_repetition,increments,dtype=int)
    mc_pricing_list_sim = []
    mc_pricing_list_direct = []

    mc_error_list_sim = []
    mc_error_list_direct = []

    for rep in tqdm.tqdm(different_mc_rep):

        pay_off_array_sim=np.zeros(rep)
        pay_off_array_direct = np.zeros(rep)

        mc_list=[mc]
        for j in range(rep):
            mc = monte_carlo(steps, T, S0, sigma, r, K)
            mc.euler_integration()
            pay_off_array_sim[j] = np.max([(mc.K-mc.euler_price_path[-1]), 0])
            pay_off_array_direct[j] = np.max([(mc.K - mc.euler_integration), 0])

        mc_mean_pay_off_sim = np.mean(pay_off_array_sim)
        mc_mean_pay_off_direct = np.mean(pay_off_array_direct)

        mc_error_list_sim.append(np.std(pay_off_array_sim)/np.sqrt(rep))
        mc_error_list_direct.append(np.std(pay_off_array_direct) / np.sqrt(rep))

        mc_pricing_list_sim.append(np.exp(-r*T)*mc_mean_pay_off_sim)
        mc_pricing_list_direct.append(np.exp(-r * T) * mc_mean_pay_off_direct)

    '''
    #saving the Data
    pickle.dump(mc_pricing_list_sim, open('Data/mc_pricing_list_sim_MC.pkl', 'wb'))
    pickle.dump(mc_pricing_list_direct, open('Data/mc_pricing_list_direct_MC.pkl', 'wb'))
    pickle.dump(mc_error_list_sim, open('Data/mc_error_list_sim_MC.pkl', 'wb'))
    pickle.dump(mc_error_list_direct, open('Data/mc_error_list_direct_MC.pkl', 'wb'))

    bs = BlackScholes(T, S0, K, r, sigma)

    bs_array = np.ones(max_repetition)*bs.put_price()

    plt.figure()
    plt.plot(different_mc_rep,mc_pricing_list_sim,color='gray',label='Monte Carlo Sim')
    plt.plot(different_mc_rep, mc_pricing_list_direct, color='k', label='Monte Carlo Direct')
    plt.plot(bs_array,'r',label='Black Scholes')
    plt.plot(different_mc_rep,mc_error_list_sim,label='Standard error Sim')
    plt.plot(different_mc_rep,mc_error_list_direct,label='Standard error Direct')
    plt.legend()
    plt.plot()
    plt.xlabel(r"MC repetition",fontsize=12,fontweight='bold')
    plt.ylabel("Option Price",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    if save_plot:
        plt.savefig("figures/"+"mc_euler_integration_diff_MC",dpi=300)
    plt.show()
    plt.close()

    '''


def diff_K_monte_carlo_process(T,K, S0, r, sigma, steps,save_plot=False):
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

    repetition = 4
    different_k = np.linspace(80,110,dtype=int)
    mc_pricing = defaultdict(list)

    for diff_strike_price in tqdm.tqdm(different_k):

        mc_list = [monte_carlo(steps, T, S0, sigma, r, diff_strike_price) for i in range(repetition)]
        num_core = 3
        pool = multiprocessing.Pool(num_core)
        pay_off_list = pool.map(worker_pay_off_euler, ((mc) for mc in mc_list))
        pool.close()
        pool.join()

        mean_pay_off_sim = np.mean([pay_off[0] for pay_off in pay_off_list])
        mean_pay_off_direct = np.mean([pay_off[1] for pay_off in pay_off_list])

        std_pay_off_sim = np.std([pay_off[0] for pay_off in pay_off_list])/repetition
        std_pay_off_direct = np.std([pay_off[1] for pay_off in pay_off_list]) / repetition

        mc_pricing['simualted_path'].append((np.exp(-r*T)*mean_pay_off_sim ,std_pay_off_sim))
        mc_pricing['direct_method'].append((np.exp(-r * T) * mean_pay_off_direct,std_pay_off_direct))

    bs_list= []
    for k in different_k:
        bs = BlackScholes(T, S0, k, r, sigma)
        bs_list.append(bs.put_price())

    fig, axs = plt.subplots(2)
    rc('text', usetex=True)
    axs[0].plot(different_k,[i[0] for i in mc_pricing['simualted_path']], color='gray', label='Monte Carlo Sim')
    axs[0].plot(different_k,[i[0] for i in mc_pricing['direct_method']], color='k', label='Monte Carlo Direct')
    axs[0].plot(different_k, bs_list, 'r', label='Black Scholes')
    axs[0].legend()
    axs[0].set_ylabel("Option Price",fontsize=14)
    axs[0].tick_params(labelsize='15')


    axs[1].plot(different_k,[i[1] for i in mc_pricing['simualted_path']],label='Simualted STD error')
    axs[1].plot(different_k, [i[1] for i in mc_pricing['direct_method']],label='Direct method STD error')
    axs[1].set_xlabel(r"Strike price $S_K$", fontsize=14)
    axs[1].legend()
    axs[1].set_ylabel("Standard error", fontsize=14)
    axs[1].tick_params(labelsize='15')


    if save_plot:
        plt.savefig("figures/" + "mc_euler_integration_diff_K", dpi=300)
    plt.show()
    plt.close()
    plt.show()



    #saving the Data
    # pickle.dump(mc_pricing_list_sim, open('Data/mc_pricing_list_sim_K.pkl', 'wb'))
    # pickle.dump(mc_pricing_list_direct, open('Data/mc_pricing_list_direct_K.pkl', 'wb'))
    # pickle.dump(mc_error_list_sim, open('Data/mc_error_list_sim_K.pkl', 'wb'))
    # pickle.dump(mc_error_list_direct, open('Data/mc_error_list_direct_K.pkl', 'wb'))


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

def bump_and_revalue(
        T, S0, K, r, sigma, steps, 
        epsilons, reps=100, full_output=False, save_plot=False, show_plots=False
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
        S_eps = S0 + eps
        for j in range(reps):

            # Create bump and revalue Monte Carlo (MC) objects
            mc_revalue = monte_carlo(steps, T, S0, sigma, r, K)
            mc_bump = monte_carlo(steps, T, S_eps, sigma, r, K)

            # Euler integration MC rev, save discounted payoff at maturity
            mc_revalue.euler_integration()
            payoff_revalue = max([mc_revalue.K - mc_revalue.euler_integration, 0])
            prices_revalue[j, i] = math.exp(-r * T) * payoff_revalue

            # Euler integration MC bump, save discounted payoff at maturity
            mc_bump.euler_integration()
            payoff_bump = max([mc_bump.K - mc_bump.euler_integration, 0])
            prices_bump[j, i] = math.exp(-r * T) * payoff_bump
 
        # Takes mean of prices bump and revalue and determines the delta 
        # for a given bump
        mean_price_revalue = prices_revalue[:, i].mean()
        mean_price_bump = prices_bump[:, i].mean()
        print("=============================================")
        print(f"REVALUE WITH EPS = {eps}")
        print(mean_price_revalue)
        print("=============================================")
        print(f"BUMP WITH EPS = {eps}")
        print(mean_price_bump)
        print("=============================================")
        deltas[i] = (mean_price_bump - mean_price_revalue) / eps

        # Determine theoretical delta for given bump
        d1 = (np.log((S_eps) / K) + (r + 0.5 * sigma ** 2)
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

