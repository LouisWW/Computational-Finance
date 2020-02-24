#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""



import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Binomial_tree import BinTreeOption, BlackScholes
import time
import multiprocessing
import math
import yfinance as yf
import statistics



def binomial_tree_1(N, T, S, K, r,market,option_type,save_plot=False):
    '''

    :param N: number of steps
    :param T: period
    :param S: stock price
    :param K: strick price
    :param r: interest rate
    :param sigma: volatility
    :param market: Eu or USA
    :return:   price of option & delta
    '''

    # Analyse various levels of volatility
    sigmas = np.linspace(0.01, 0.99, 100)
    trees= [
     BinTreeOption(N, T, S, s, r, K, market, option_type)
     for s in sigmas
     ]
    bs = [BlackScholes(T, S, K, r, s) for s in sigmas]

    call_prices = defaultdict(list)
    for tree, bs in zip(trees, bs):
     call_prices["Binomial tree"].append(tree.determine_price())
     call_prices["Black Scholes"].append(bs.call_price())

    # # Make plot
    plt.figure()
    plt.plot(sigmas, [i[0] for i in call_prices["Binomial tree"]], label="Binomial tree")
    plt.plot(sigmas, call_prices["Black Scholes"], label="Black Scholes")
    plt.xlabel("Volatility (%) ",fontsize=12,fontweight='bold')
    plt.ylabel("Price",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title(market+" "+option_type+" option price for various levels of volatility",fontsize=14,fontweight='bold')
    plt.legend()
    if save_plot:
        plt.savefig("figures/"+market+"_"+option_type+"_volatility",dpi=300)
    plt.show()
    plt.close()


def worker(tree):

    return tree.determine_price()




def binomial_tree_2( T, S, K, r, sigma, market, option_type,save_plot=False,run_time=True):
    '''
    :param T: period
    :param S: stock price
    :param K: strick price
    :param r: interest rate
    :param sigma: volatility
    :param market: Eu or USA
    :return:   price of option & delta
    '''

    # # Analyse time steps

    steps = list(range(20, 500,5))

    trees = [
        BinTreeOption(step, T, S, sigma, r, K, market, option_type)
        for step in steps
    ]

    NUM_CORE = 2
    pool = multiprocessing.Pool(NUM_CORE)
    prices_trees = pool.map(worker, ((tree) for tree in trees))
    pool.close()
    pool.join()

    bs = BlackScholes(T, S, K, r, sigma)

    if option_type=='call':
        bs_price = bs.call_price()
    else:
        bs_price = bs.put_price()

    print("Black Scholes option price =",bs_price)
    prices_bs = [bs_price] * len(steps)

    # # Make plot
    plt.figure()
    plt.plot(steps, [i[0] for i in prices_trees], label="Binomial tree")
    plt.plot(steps, prices_bs, label="Black Scholes")
    plt.xlabel("Time steps (a.u.)",fontsize=12,fontweight='bold')
    plt.ylabel("Price",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title(market+" "+option_type+" option price for increasing time steps",fontsize=14,fontweight='bold')
    plt.legend()
    if save_plot:
        plt.savefig("figures/"+market+"_"+option_type+"_time_steps",dpi=300)

    # Get the running time
    if run_time:
        repetition = 20
        running_time_matrix = np.zeros((len(steps) + 1, repetition))
        steps = list(range(1, 100))
        for i in range(repetition):
            for step in steps:
                start_time=time.time()
                tree =BinTreeOption(step, T, S, sigma, r, K, market, option_type)
                running_time=(time.time()-start_time)*100
                running_time_matrix[step][i]=running_time

        mean_running_time=np.mean(running_time_matrix,1)
        mean_running_time=np.delete(mean_running_time,0)

        plt.figure()
        plt.plot(steps, mean_running_time, label="Running time")
        plt.xlabel("Time steps (a.u.)",fontsize=12,fontweight='bold')
        plt.ylabel("Running Time (ms)",fontsize=12,fontweight='bold')
        plt.legend()
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.title("Running time vs. Steps",fontsize=14,fontweight='bold')
        if save_plot:
            plt.savefig("figures/"+market+"_"+option_type+"_running_time",dpi=300)

        plt.show()
        plt.close()


def binomial_tree_3(N,T, S, K, r, market, option_type,save_plot=True):
    '''

    :param N: number of steps
    :param T: period
    :param S: stock price
    :param K: strick price
    :param r: interest rate
    :param sigma: volatility
    :param market: Eu or USA
    :return:   price of option & delta
    '''

    # Analyse various levels of volatility
    sigmas = np.linspace(0.01, 0.99, 100)
    trees = [
        BinTreeOption(N, T, S, s, r, K, market, option_type)
        for s in sigmas
    ]
    bs_list = [BlackScholes(T, S, K, r, s) for s in sigmas]

    call_prices = defaultdict(list)
    for tree, bs in zip(trees, bs_list):
        call_prices["Binomial tree"].append(tree.determine_price())

    #  Make plot
    plt.figure()
    plt.plot(sigmas, [i[1] for i in call_prices["Binomial tree"]], label="Binomial tree")
    plt.plot(sigmas, [i[2] for i in call_prices["Binomial tree"]], label="Black Scholes")
    plt.xlabel("Volatility (%) ",fontsize=12,fontweight='bold')
    plt.ylabel(r"$\Delta$ (%)",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title(market+" "+option_type+r" $\Delta$ for various levels of volatility",fontsize=14,fontweight='bold')
    plt.legend()
    if save_plot:
        plt.savefig("figures/"+market+"_"+option_type+"_volatility_delta",dpi=300)
    plt.show()
    plt.close()


def wiener_process(T, S0, K, r, sigma, steps=1,save_plot=True):
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
    bs = BlackScholes(1, 100, 99, 0.06, 0.2, steps=365)
    bs.create_price_path()

    plt.figure()
    plt.plot(bs.price_path)
    plt.xlabel("Days",fontsize=12,fontweight='bold')
    plt.ylabel("Stock price",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title("Stock price simulated based on the Wiener process",fontsize=14,fontweight='bold')
    if save_plot:
        plt.savefig("figures/"+"wiener_process",dpi=300)
    plt.show()
    plt.close()


def real_stock_data():
    years = 1
    rate = 0.06

    def fill_year(data, open_close='Open'):

        if open_close == "Open":
            time_serie = data.Open
        elif open_close == 'Close':
            time_serie = data.Close
        else:
            print(open_close, 'is not knows')
            return None

        n_days_in_years = 365
        days = np.zeros(n_days_in_years)

        i = 0
        s = 0

        for start, timestamp in enumerate(time_serie.index):
            if timestamp.weekday() == 0:
                break

        for value in time_serie[start:]:
            days[i] = value
            i += 1
            s += 1
            if s % 5 == 0:
                days[i] = value
                days[i + 1] = value
                i += 2

            if i == 365:
                break

        return days

    def get_data(stock='AAPL', frm='2019-01-01', till='2020-02-01'):
        data = yf.download(stock, frm, till)
        return data

    def get_implied_volatility(data):
        volatility = np.std(data) / np.mean(data)
        return volatility

    def plot_price_path(B, title, hedge_plot=True):
        fig, ax1 = plt.subplots()
        x_price = [i / B.steps for i in range(B.steps)]
        x_price = [i for i in range(1, 366)]

        color = 'tab:red'
        ax1.set_xlabel('Days',fontsize=12,fontweight='bold')
        ax1.set_ylabel('Price', color=color,fontsize=12,fontweight='bold')
        ax1.plot(x_price, B.price_path, color=color,
                 label="Discritized Black Scholes")
        ax1.tick_params(axis='y', labelcolor=color)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        if hedge_plot:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            print('plot2')
            # we already handled the x-label with ax1
            ax2.set_ylabel('Delta', color=color,fontsize=12,fontweight='bold')
            ax2.scatter(B.x_hedge, B.delta_list, color=color, label='Hedge delta')
            ax2.plot(B.x_hedge, B.delta_list, linestyle='--', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            plt.yticks(fontweight='bold')

        plt.title(title,fontsize=14,fontweight='bold')
        plt.tight_layout()
        plt.savefig("figures/"+title+'.png',dpi=300)

    data = fill_year(get_data(stock='AAPL'), open_close='Open')
    sigma = get_implied_volatility(data)
    steps = 365

    B = BlackScholes(years, data[0], data[0] - 1, rate, sigma, steps)
    B.price_path = data
    print('profit of Apple stocks:', B.create_hedge(52,hedge_setting='call'))

    B.x_hedge = [i * 7 for i in range(0, 52)]
    plot_price_path(B, 'Apple stocks simulation')

    data = fill_year(get_data(stock='RDS-A'), open_close='Open')
    sigma = get_implied_volatility(data)

    B = BlackScholes(years, data[0], data[0] - 1, rate, sigma, steps)
    B.price_path = data
    print('profit of Shell simuation:', B.create_hedge(52))

    B.x_hedge = [i * 7 for i in range(0, 52)]
    plot_price_path(B, 'Shell stocks simulation')


def profit_histogram():
    steps = 365
    years = 1
    start_price = 100
    strike_price = 99
    rate = 0.06
    volatility = 0.2

    prof=np.zeros(1000)
    for i in range(1000):
        B = BlackScholes(years, start_price, strike_price, rate, volatility, steps)
        prof[i] = B.create_hedge(steps)

    print("Daily: Standard Deviation ",statistics.stdev(prof))
    fig = plt.figure()
    plt.hist(prof, bins=20, label=f"with mean: {round(np.mean(prof), 3)}")
    plt.xlabel('Profit',fontsize=12,fontweight='bold')
    plt.ylabel('Frequency',fontsize=12,fontweight='bold')
    plt.title('Hedging delta every day',fontsize=14,fontweight='bold')
    plt.legend()
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    fig.savefig('figures/hedgedeltaday.png', dpi=300)

    steps = 52
    prof= np.zeros(1000)
    for i in range(1000):
        B = BlackScholes(years, start_price, strike_price, rate, volatility, steps)
        prof[i] = B.create_hedge(steps)

    print("Weekly: Standard Deviation ", statistics.stdev(prof))
    fig = plt.figure()
    plt.hist(prof, bins=20, label=f"with mean: {round(np.mean(prof), 3)}")
    plt.xlabel('Profit',fontsize=12,fontweight='bold')
    plt.ylabel('Frequency',fontsize=12,fontweight='bold')
    plt.title('Hedging delta every week',fontsize=14,fontweight='bold')
    plt.legend()
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    fig.savefig('figures/hedgedeltaweek.png', dpi=300)


def all_profit_histograms():
    price_steps = 365
    years = 1
    start_price = 100
    strike_price = 99
    rate = 0.06
    volatility = 0.2
    fig = plt.figure()

    steps_array = []
    for steps in [10, 50, 100, 200, 300]:
        prof = np.zeros(1000)
        for i in range(1000):
            B = BlackScholes(years, start_price, strike_price, rate, volatility, price_steps)
            prof[i] = B.create_hedge(steps)

        print("Steps ",steps,": Standard Deviation: ",statistics.stdev(prof))
        plt.hist(prof, bins=20, label=f'n={steps}')
        steps_array.append(np.mean(prof))

    plt.xlabel('Profit',fontsize=12,fontweight='bold')
    plt.ylabel('Frequency',fontsize=12,fontweight='bold')
    plt.title('Hedging delta different intervals',fontsize=14,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend()
    plt.tight_layout()

    fig.savefig('figures/different_steps.png')
