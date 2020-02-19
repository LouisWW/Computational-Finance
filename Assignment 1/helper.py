import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Binomial_tree import BinTreeOption, BlackScholes
import time
import tqdm
import multiprocessing

global progress_bar




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



steps = list(range(20, 500,5))
progress_bar = tqdm.tqdm(total=len(steps))

def worker(tree):
    progress_bar.update(1)
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

    parralel_time=time.time()


    trees = [
        BinTreeOption(step, T, S, sigma, r, K, market, option_type)
        for step in steps
    ]
    NUM_CORE = 3
    pool = multiprocessing.Pool(NUM_CORE)
    prices_trees = pool.map(worker, ((tree) for tree in trees))
    pool.close()
    pool.join()

    print("parallel time ",time.time()-parralel_time)




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
        plt.savefig("figures/"+market+"_"+option_type+"_time_steps",dpi=300)


    ### Get the running ###

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



    # # Make plot
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
