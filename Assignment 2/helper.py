#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

from monte_carlo import monte_carlo
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import multiprocessing


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


def monte_carlo_process(T, S0, K, r, sigma, steps,save_plot=False):
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

    rep = 100
    sub_rep= 1000
    mean_pay_off_array = np.zeros(rep)
    for i in range(rep):

        mc_list = [monte_carlo(steps, T, S0, sigma, r, K) for i in range(sub_rep)]
        pay_off_array = np.zeros(sub_rep)



        for j in range(sub_rep):
            mc_list[j].wiener_method()
            pay_off_array[j] = np.max([(mc_list[j].wiener_price_path[-1]-mc_list[j].K),0])


        mean_pay_off_array[i] = np.mean(pay_off_array)

    std_dev=np.std(mean_pay_off_array)

    print("The standard deviation is ", std_dev)

    plt.figure()
    plt.hist(mean_pay_off_array,color='gray')
    plt.plot()
    plt.xlabel(r"Stock price S_T",fontsize=12,fontweight='bold')
    plt.ylabel("Occurence (#)",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    if save_plot:
        plt.savefig("figures/"+"mc_on_wiener_process",dpi=300)
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


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
def test(T, S0, K, r, sigma, steps,save_plot=False):
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

    rep = 3
    sub_rep= 10
    mean_pay_off_array = np.zeros(rep)
    for i in range(rep):
        mc_list =[]
        mc_list = [monte_carlo(steps, T, S0, sigma, r, K) for i in range(sub_rep)]
        pay_off_array = np.zeros(sub_rep)

        NUM_CORE = 2
        pool = multiprocessing.Pool(NUM_CORE)
        prices_trees = pool.map(worker, ((mc) for mc in mc_list))
        pool.close()
        pool.join()


        print(prices_trees)
