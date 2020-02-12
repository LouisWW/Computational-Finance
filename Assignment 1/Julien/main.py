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
