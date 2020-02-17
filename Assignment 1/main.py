'''
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from Binomial_tree import BinTreeOption, BlackScholes


if __name__ == "__main__":

    # # Test estimations
    # N, T, S, K, r, sigma = 50, 1, 100, 99, 0.06, 0.2
    # tree_call_eu = BinTreeOption(N, T, S, sigma, r, K, market="EU", option_type="call")
    # bs_eu = BlackScholes(T, S, K, r, sigma)

    # tree_call_price = tree_call_eu.determine_price()
    # bs_call_price = bs_eu.call_price()
    # print("Binomial Call Price: ", tree_call_price)
    # print("Black Scholes Call Price: ", bs_call_price)

    # tree_put_eu = BinTreeOption(N, T, S, sigma, r, K, market="EU", option_type="put")
    # tree_put_price = tree_put_eu.determine_price()
    # bs_put_price = bs_eu.put_price()
    # print("Binomial Put Price: ", tree_put_price)
    # print("Black Scholes Put Price: ", bs_put_price)

    # # Analyse various levels of volatility
    # sigmas = np.linspace(0.01, 0.99, 100)
    # trees_call_eu = [
    #     BinTreeOption(N, T, S, s, r, K, market="EU", option_type="call")
    #     for s in sigmas
    #     ]
    # bs_eus = [BlackScholes(T, S, K, r, s) for s in sigmas]
    
    # call_prices = defaultdict(list)
    # for tree, bs in zip(trees_call_eu, bs_eus):
    #     call_prices["Binomial tree"].append(tree.determine_price())
    #     call_prices["Black Scholes"].append(bs.call_price())

    # # Make plot
    # fig = plt.figure()
    # plt.plot(sigmas, call_prices["Binomial tree"], label="Binomial tree")
    # plt.plot(sigmas, call_prices["Black Scholes"], label="Black Scholes")
    # plt.xlabel("Volatility")
    # plt.ylabel("Price")
    # plt.title("European call option price for various levels of volatility")
    # plt.legend()
    # plt.show()
    # plt.close()


    # # Analyse time steps
    # steps = list(range(1, 100))
    # trees_call_eu = [
    #     BinTreeOption(step, T, S, sigma, r, K, market="EU", option_type="call")
    #     for step in steps
    #     ]

    # prices_trees = [tree.determine_price() for tree in trees_call_eu]
    # prices_bs = [bs_call_price] * len(steps)
    
    # # Make plot
    # fig = plt.figure()
    # plt.plot(steps, prices_trees, label="Binomial tree")
    # plt.plot(steps, prices_bs, label="Black Scholes")
    # plt.xlabel("Time steps")
    # plt.ylabel("Price")
    # plt.title("European call option price for increasing time steps")
    # plt.legend()
    # plt.show()
    # plt.close()


    bs_eu_daily = BlackScholes(1, 100, 99, 0.06, 0.2, steps=365)
    bs_eu_daily.create_price_path()
    bs_eu_daily.plot_price_path(hedge_setting="Call", hedge_plot=True, steps=365)
    bs_eu_daily.plot_price_path(hedge_setting="Call", hedge_plot=True, steps=52)
    plt.show()
'''



import helper as helper


# exercise 1.1
# helper.binomial_tree_1(50, 1, 100, 99, 0.06,'EU',"call",save_plot=True)
#helper.binomial_tree_1(50, 1, 100, 99, 0.06,'EU',"put",save_plot=True)

# exercise 1.2
#helper.binomial_tree_2(1, 100, 99, 0.06, 0.2,'EU',"call",save_plot=True)


# exercise 1.3
helper.binomial_tree_3(50,1, 100, 99, 0.06,'EU',"call",save_plot=True)

# exercise 1.4 is the same as 1.1 with different arguments
#helper.binomial_tree_1(50, 1, 100, 99, 0.06,0.2,'USA',"put",save_plot=True)
#helper.binomial_tree_2(1, 100, 99, 0.06, 0.2,'USA',"put",save_plot=True)
