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
import tqdm
from collections import defaultdict
import multiprocessing
from Binomial_tree import BinTreeOption, BlackScholes
import tqdm



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
        for j in range(rep):
            mc = monte_carlo(steps, T, S0, sigma, r, K)
            mc.euler_integration()
            pay_off_array_sim[j] = np.max([(mc.K-mc.euler_price_path[-1]), 0])
            pay_off_array_direct[j] = np.max([(mc.K - mc.euler_integration), 0])

        mc_mean_pay_off_sim = np.mean(pay_off_array_sim)
        mc_mean_pay_off_direct = np.mean(pay_off_array_direct)

        mc_error_list_sim.append(np.std(pay_off_array_sim)/np.sqrt(rep))
        mc_error_list_direct.append(np.std(pay_off_array_direct) / np.sqrt(rep))

<<<<<<< HEAD
        for j in range(repetition):
            mc_list[j].euler_integration()
            mc_pay_off_array[j] = np.max([(mc_list[j].K-mc_list[j].euler_price_path[-1]), 0])
=======
        mc_pricing_list_sim.append(np.exp(-r*T)*mc_mean_pay_off_sim)
        mc_pricing_list_direct.append(np.exp(-r * T) * mc_mean_pay_off_direct)
>>>>>>> e4729a90202b016d7ce1ec90d360ac593d94d699


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



    max_repetition = 10000
    different_K = np.linspace(80,110,dtype=int)
    mc_pricing_list_sim = []
    mc_pricing_list_direct = []

    mc_error_list_sim = []
    mc_error_list_direct = []

    for diff_strike_price in tqdm.tqdm(different_K):

        pay_off_array_sim=np.zeros(max_repetition)
        pay_off_array_direct = np.zeros(max_repetition)
        for j in range(max_repetition):
            mc = monte_carlo(steps, T, S0, sigma, r, diff_strike_price)
            mc.euler_integration()
            pay_off_array_sim[j] = np.max([(mc.K-mc.euler_price_path[-1]), 0])
            pay_off_array_direct[j] = np.max([(mc.K - mc.euler_integration), 0])

        mc_mean_pay_off_sim = np.mean(pay_off_array_sim)
        mc_mean_pay_off_direct = np.mean(pay_off_array_direct)

        mc_error_list_sim.append(np.std(pay_off_array_sim)/np.sqrt(max_repetition))
        mc_error_list_direct.append(np.std(pay_off_array_direct) / np.sqrt(max_repetition))

        mc_pricing_list_sim.append(np.exp(-r*T)*mc_mean_pay_off_sim)
        mc_pricing_list_direct.append(np.exp(-r * T) * mc_mean_pay_off_direct)


    bs_list=[]
    for k in different_K:
        bs = BlackScholes(T, S0, k, r, sigma)
        bs_list.append(bs.put_price())



    plt.figure()
    plt.plot(different_K,mc_pricing_list_sim,color='gray',label='Monte Carlo Sim')
    plt.plot(different_K, mc_pricing_list_direct, color='k', label='Monte Carlo Direct')
    plt.plot(different_K,bs_list,'r',label='Black Scholes')
    plt.plot(different_K,mc_error_list_sim,label='Standard error Sim')
    plt.plot(different_K,mc_error_list_direct,label='Standard error Direct')
    plt.legend()
    plt.plot()
    plt.xlabel(r"Strike price S_K",fontsize=12,fontweight='bold')
    plt.ylabel("Option Price",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    if save_plot:
        plt.savefig("figures/"+"mc_euler_integration_diff_K",dpi=300)
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

def bump_and_revalue(T, S0, K, r, sigma, steps, epsilons, save_plot=False):
    """
    """
    repetitions, diff_eps = 100, len(epsilons)
    prices_revalue = np.zeros((repetitions, diff_eps))
    prices_bump = np.zeros((repetitions, diff_eps))
    deltas = np.zeros(diff_eps)
    bs_deltas = np.zeros(diff_eps)

    for i, eps in enumerate(epsilons):
        mc_revalue_list = [monte_carlo(steps, T, S0, sigma, r, K)
                           for _ in range(repetitions)]
        mc_bump_list = [monte_carlo(steps, T, S0 + eps, sigma, r, K) 
                        for _ in range(repetitions)]
        # payoff_revalue = np.zeros(repetitions)
        # payoff_bump = np.zeros(repetitions)

        for j in range(repetitions):
            mc_revalue_list[j].euler_integration()
            payoff_revalue = max([mc_revalue_list[j].K - 
                                    mc_revalue_list[j].euler_price_path[-1], 0])
            prices_revalue[j, i] = math.exp(-r * T) * payoff_revalue
            #print(payoff_revalue)

            mc_bump_list[j].euler_integration()
            payoff_bump = max([mc_bump_list[j].K - 
                                mc_bump_list[j].euler_price_path[-1], 0])
            prices_bump[j, i] = math.exp(-r * T) * payoff_bump
            #print(payoff_bump)

            #deltas[j, i] = (payoff_bump - payoff_revalue) / eps
            #(deltas[j, :])
            #break

        #break
        mean_price_revalue = prices_revalue[:, i].mean()
        mean_price_bump = prices_bump[:, i].mean()
        print(prices_revalue[:, i])
        print(mean_price_revalue)
        print(prices_bump[:, i])
        print(mean_price_bump)
        deltas[i] = (mean_price_bump - mean_price_revalue) / eps
        break

    return deltas

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
