import numpy as np

def binom_tree(N, T, S0, sigma, r, K, market="EU", option_type="call",
                array_out=False):

    # Init
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Price tree
    price_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            price_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Option value
    option = np.zeros((N + 1, N + 1))

    # Calculate option price at t=0 (European)
    discount = np.exp(-r * dt)
    if market == "EU" and option_type == "call":
        option[:, N] = np.maximum(np.zeros(N + 1), price_tree[:, N] - K)
        recursive_eu_option(option, N, discount, p)

    elif market == "EU" and option_type == "put":
        option[:, N] = np.maximum(np.zeros(N + 1), K - price_tree[:, N])
        recursive_eu_option(option, N, discount, p)

    elif market == "USA" and option_type == "call":
        option[:, N] = np.maximum(np.zeros(N + 1), price_tree[:, N] - K)
        recursive_usa_call(option, price_tree, K, N, discount, p)

    elif market == "USA" and option_type == "put":
        option[:, N] = np.maximum(np.zeros(N + 1), K - price_tree[:, N])
        recursive_usa_put(option, price_tree, K, N, discount, p)

    if array_out:
        return [option[0, 0], price_tree, option]
    else:
        return option[0, 0]

def recursive_eu_option(option, N, discount, p):
    """
    """
    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i + 1):
            option[j, i] = discount * (p * option[j, i + 1] +
                                                (1 - p) * option[j + 1, i + 1])

def recursive_usa_call(option, price_tree, K, N, discount, p):
    """
    """
    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i + 1):
            paths = [
                0,
                price_tree[j, i] - K,
                discount * (p * option[j, i + 1] +
                            (1 - p) * option[j + 1, i + 1])
            ]
            option[j, i] = max(paths)

def recursive_usa_put(option, price_tree, K, N, discount, p):

    for i in np.arange(N - 1, -1, -1):
        for j in np.arange(0, i + 1):
            paths = [
                0,
                K - price_tree[j, i],
                discount * (p * option[j, i + 1] +
                            (1 - p) * option[j + 1, i + 1])
            ]
            option[j, i] = max(paths)
            

if __name__ == "__main__":
    # print("=========================================================")
    # op_price, price_tree, option = binom_tree(
    #     5, 5 / 12, 50, 0.4, 0.1, 50, market="USA", option_type="put", array_out=True)
    # print("=========================================================")
    # print(op_price)
    # print(price_tree)
    # print(option)
    # print("=========================================================")

    # op_price, price_tree, option = binom_tree(
    #     5, 5 / 12, 50, 0.4, 0.1, 50, market="EU", option_type="put", array_out=True)
    # print("=========================================================")
    # print(op_price)
    # print(price_tree)
    # print(option)
    # print("=========================================================")

    # op_price, price_tree, option = binom_tree(50, 1, 100, 0.2, 0.06, 99, market="USA", option_type="call", array_out=True)
    # print(op_price)
    # print(price_tree)
    # print(option)
    # print("=========================================================")
    # op_price, price_tree, option = binom_tree(50, 1, 100, 0.2, 0.06, 99, market="EU", option_type="call", array_out=True)
    # print(op_price)
    # print(price_tree)
    # print(option)
    # print("=========================================================")


    op_price, price_tree, option = binom_tree(
        50, 1, 100, 0.2, 0.06, 99, market="USA", option_type="put", array_out=True)
    print("=========================================================")
    print(op_price)
    print(price_tree)
    print(option)

    print("=========================================================")

    op_price, price_tree, option = binom_tree(
        50, 1, 100, 0.2, 0.06, 99, market="EU", option_type="put", array_out=True)
    print("=========================================================")
    print(op_price)
    print(price_tree)
    print(option)
    print("=========================================================")
    pass