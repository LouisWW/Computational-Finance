import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Binomial_tree import BinTreeOption, BlackScholes
import time

def binomial_tree_1(N, T, S, K, r, sigma,market,option_type,save_plot=False):
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
    trees_call_eu = [
     BinTreeOption(N, T, S, s, r, K, market, option_type)
     for s in sigmas
     ]
    bs_eus = [BlackScholes(T, S, K, r, s) for s in sigmas]

    call_prices = defaultdict(list)
    for tree, bs in zip(trees_call_eu, bs_eus):
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


def binomial_tree_2( T, S, K, r, sigma, market, option_type,save_plot=False):
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
    trees=[]
    running_time=[]
    steps = list(range(1, 100))
    for step in steps:
        start_time=time.time()
        trees.append(BinTreeOption(step, T, S, sigma, r, K, market, option_type))
        running_time.append((time.time()-start_time)*100)

    prices_trees = [tree.determine_price() for tree in trees]

    bs = BlackScholes(T, S, K, r, sigma)
    if option_type=='call':
        bs_price = bs.call_price()
    else:
        bs_price = bs.put_price()

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
        plt.savefig("figures/"+market+"_"+option_type+"_running_time",dpi=300)


    plt.figure()
    plt.plot(steps, running_time, label="Running time")
    plt.xlabel("Time steps (a.u.)",fontsize=12,fontweight='bold')
    plt.ylabel("Running Time (ms)",fontsize=12,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.title("Running time vs. Steps",fontsize=14,fontweight='bold')

    plt.show()
    plt.close()

def binomial_tree_3(T, S, K, r, sigma, market, option_type, save_plot=False):
    tree=BinTreeOption(step, T, S, sigma, r, K, market, option_type)
