import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from test_code_julien import BinTreeOption, BlackScholes


if __name__ == "__main__":
    N, T, S, K, r, sigma = 50, 1, 100, 99, 0.06, 0.2
    tree_call_eu = BinTreeOption(N, T, S, sigma, r, K, market="EU", option_type="call")
    bs_eu = BlackScholes(T, S, K, r, sigma)

    tree_call_price = tree_call_eu.determine_price()
    bs_call_price = bs_eu.call_price()
    print("Binomial Call Price: ", tree_call_price)
    print("Black Scholes Call Price: ", bs_call_price)

    tree_put_eu = BinTreeOption(N, T, S, sigma, r, K, market="EU", option_type="put")
    tree_put_price = tree_put_eu.determine_price()
    bs_put_price = bs_eu.put_price()
    print("Binomial Put Price: ", tree_put_price)
    print("Black Scholes Put Price: ", bs_put_price)

    sigmas = np.linspace(0.01, 0.99, 100)
    trees_call_eu = [
        BinTreeOption(N, T, S, s, r, K, market="EU", option_type="call")
        for s in sigmas
        ]
    bs_eus = [BlackScholes(T, S, K, r, s) for s in sigmas]
    
    call_prices = defaultdict(list)
    for tree, bs in zip(trees_call_eu, bs_eus):
        call_prices["Binomial tree"].append(tree.determine_price())
        call_prices["Black Scholes"].append(bs.call_price())

    fig = plt.figure()
    plt.plot(sigmas, call_prices["Binomial tree"], label="Binomial tree")
    plt.plot(sigmas, call_prices["Black Scholes"], label="Black Scholes")
    plt.xlabel("Volatility")
    plt.ylabel("Price")
    plt.title("European call option price for various levels of volatility")
    plt.legend()
    plt.show()
    plt.close()
