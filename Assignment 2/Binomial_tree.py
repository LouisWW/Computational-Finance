#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

# Import built-in libs
import math

# Import 3th parties libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

class BinTreeOption:
    def __init__(
        self, N, T, S0, sigma, r, K,
        market="EU", option_type="call", array_out=False
    ):
        """
        OOP representation of a binomial option tree.
        Input:
            N = total time steps (integer)
            T =  maturity option in years (numeric)
            S0 = initial stock price (numeric)
            r = risk-free rate (numeric)
            K = strike price option (numeric)
            market = market type (EU or USA)
            option_type = determines option type (call or put)
            array_out = False gives only resulting values, True gives full trees
        Output:
            returns an object representation with an already created price 
            tree. It also contains methods to determine option price 
            development and the hedging strategy
        """

        # Init
        self.N = N
        self.T = T
        self.S0 = S0
        self.sigma = sigma
        self.r = r
        self.K = K
        self.market = market.upper()
        self.option_type = option_type.lower()
        self.array_out = array_out

        # Checks if market type and option type are valid
        assert self.market in ["EU", "USA"], "Market not found. Choose EU or USA"
        assert self.option_type in ["call", "put"], "Non-existing option type."

        # Setup parameters for movements binomial tree
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)

        # Create price tree and initialize option tree
        self.price_tree = np.zeros((N + 1, N + 1))
        self.create_price_tree()
        self.option = np.zeros((N + 1, N + 1))

        # Create hedging tree and theoretical hedging tree
        self.delta = np.zeros((N, N))
        self.t_delta = np.zeros((N, N))

    def create_price_tree(self):
        """
        Determines stock price at every time step.
        """
        for i in range(self.N + 1):
            for j in range(i + 1):
                self.price_tree[j, i] = self.S0 * \
                    (self.u ** (i - j)) * (self.d ** j)

    def determine_price(self):
        """
        Determines option price and hedging strategy at every time step 
        depending on the option type and market.
        """

        # Sets option price at maturity and apply recursive scheme 
        # for European call option
        if self.market == "EU" and self.option_type == "call":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.price_tree[:, self.N] - self.K
            )
            self.recursive_eu_call()

        # Sets option price at maturity and apply recursive scheme
        # for European put option
        elif self.market == "EU" and self.option_type == "put":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.K - self.price_tree[:, self.N]
            )
            self.recursive_eu_put()

        # Sets option price at maturity and apply recursive scheme
        # for American call option
        elif self.market == "USA" and self.option_type == "call":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.price_tree[:, self.N] - self.K
            )
            self.recursive_usa_call()

        # Sets option price at maturity and apply recursive scheme
        # for American put option
        elif self.market == "USA" and self.option_type == "put":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.K - self.price_tree[:, self.N])
            self.recursive_usa_put()

        # Ensures full output is given if asked by user. 
        # Otherwise it only returns the variables of interest at the spot time.
        if self.array_out and self.market == "EU":
            return [self.option[0, 0], self.delta[0, 0], self.t_delta[0, 0],
                    self.price_tree, self.option, self.delta, self.t_delta]
        elif self.array_out and self.market == "USA":
            return [self.option[0, 0], self.delta[0, 0], 
                    self.price_tree, self.option, self.delta]
        elif not self.array_out and self.market == "EU":
            return self.option[0, 0], self.delta[0, 0], self.t_delta[0, 0]
        
        return self.option[0, 0], self.delta[0, 0]

    def recursive_eu_call(self):
        """
        Recursive scheme for an Europen call option.
        """

        # Time starts at maturity (only necessary for theoretical hedging)
        t = self.T

        # Start scheme
        for i in np.arange(self.N - 1, -1, -1):
            t -= self.dt

            # Determines option price, hedging strategy and theoretical hedging 
            # strartegy for each node in current layer
            for j in np.arange(0, i + 1):
                self.option[j, i] = (self.discount * (self.p *
                                                      self.option[j, i + 1] + (1 - self.p) *
                                                      self.option[j + 1, i + 1]))

                self.delta[j, i] = ((self.option[j, i + 1] -
                                     self.option[j + 1, i + 1]) /
                                    (self.price_tree[j, i + 1] -
                                     self.price_tree[j + 1, i + 1]))
                d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
                self.t_delta[j, i] = st.norm.cdf(d1, 0.0, 1.0)
                
    def recursive_eu_put(self):
        """
        Recursive scheme for an Europen put option.
        """

        # Time starts at maturity (only necessary for theoretical hedging)
        t = self.T
        
        # Start scheme
        for i in np.arange(self.N - 1, -1, -1):
            t -= self.dt

            # Determines option price, hedging strategy and theoretical hedging
            # strartegy for each node in current layer
            for j in np.arange(0, i + 1):
                self.option[j, i] = (self.discount * (self.p *
                                                      self.option[j, i + 1] + (1 - self.p) *
                                                      self.option[j + 1, i + 1]))

                self.delta[j, i] = ((self.option[j, i + 1] -
                                     self.option[j + 1, i + 1]) /
                                    (self.price_tree[j, i + 1] -
                                     self.price_tree[j + 1, i + 1]))
                
                d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
                self.t_delta[j, i] = -st.norm.cdf(-d1, 0.0, 1.0)

    def recursive_usa_call(self):
        """
        Recursive scheme for an American call option
        """

        # Start scheme
        for i in np.arange(self.N - 1, -1, -1):

            # Determines option price and hedging strategy
            # for each node in current layer
            for j in np.arange(0, i + 1):
                self.option[j, i] = max([0, self.price_tree[j, i] - self.K,
                                         self.discount *
                                         (self.p * self.option[j, i + 1] +
                                          (1 - self.p) * self.option[j + 1, i + 1])])

                self.delta[j, i] = ((self.option[j, i + 1] -
                                     self.option[j + 1, i + 1]) /
                                    (self.price_tree[j, i + 1] -
                                     self.price_tree[j + 1, i + 1]))

    def recursive_usa_put(self):
        """
        Recursive scheme for an American put option
        """

        # Start scheme
        for i in np.arange(self.N - 1, -1, -1):

            # Determines option price and hedging strategy
            # for each node in current layer
            for j in np.arange(0, i + 1):
                self.option[j, i] = max([0, self.K - self.price_tree[j, i],
                                         self.discount *
                                         (self.p * self.option[j, i + 1] +
                                          (1 - self.p) * self.option[j + 1, i + 1])])
                self.delta[j, i] = ((self.option[j, i + 1] -
                                     self.option[j + 1, i + 1]) /
                                    (self.price_tree[j, i + 1] -
                                     self.price_tree[j + 1, i + 1]))
                                     


class BlackScholes:
    def __init__(self, T, S0, K, r, sigma, steps=1):
        self.T = T
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.dt = T / steps
        self.price = S0
        self.price_path = np.zeros(steps)

        self.delta_list = None
        self.x_hedge = None

    def call_price(self, t=0):
        """
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.sigma * np.sqrt(self.T - t)

        call = (self.S0 * st.norm.cdf(d1, 0.0, 1.0) - self.K *
                np.exp(-self.r * self.T) * st.norm.cdf(d2, 0.0, 1.0))

        return call

    def put_price(self, t=0):
        """
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.sigma * np.sqrt(self.T - t)

        put = ((self.K * np.exp(-self.r * self.T)
                * st.norm.cdf(-d2, 0.0, 1.0)) - self.S0 *
               st.norm.cdf(-d1, 0.0, 1.0))

        return put

    def asian_call_price(self, t=0) :
        """
        """
        N = self.steps
        sigma = self.sigma * np.sqrt(((N + 1) * (2 * N + 1)) / (6 * N ** 2))
        b = ((N + 1) / (2 * N)) * (self.r - 0.5 * (sigma ** 2))

        d1 = ((np.log(self.K / self.S0) + (b + 0.5 * self.sigma ** 2) * (self.T - t)) /
              (sigma * np.sqrt(self.T - t)))

        d2 = d1 - sigma * np.sqrt(self.T - t)

        call = (self.S0 * np.exp((b - self.r) * self.T) * st.norm.cdf(d1, 0.0, 1.0) - self.K *
                np.exp(-self.r * self.T) * st.norm.cdf(d2, 0.0, 1.0))

        return call

    def asian_put_price(self, t=0) :
        """
        """
        N = self.steps
        sigma = self.sigma * np.sqrt(((N + 1) * (2 * N + 1)) / (6 * N ** 2))
        b = ((N + 1) / (2 * N)) * (self.r - 0.5 * (sigma ** 2))

        d1 = ((np.log(self.K / self.S0) + (b + 0.5 * sigma ** 2) * (self.T - t)) /
              (sigma * np.sqrt(self.T - t)))

        d2 = d1 - (sigma * np.sqrt(self.T - t))

        put = self.K * np.exp(-self.r * self.T) * st.norm.cdf(-d2, 0.0, 1.0) - (
                    self.S0 * np.exp((b - self.r) * self.T) * st.norm.cdf(-d1, 0.0, 1.0))
        return put

    def create_price_path(self):
        """
        """
        for i in range(self.steps):
            self.price_path[i] = self.price
            dS = self.r * self.price * self.dt + self.sigma * \
                self.price * np.random.normal(0, 1) * np.sqrt(self.dt)

            self.price += dS

    def create_hedge(self, steps=1, hedge_setting='Call'):
        '''
        Simulate hedging over the given price path and returns a profit
        '''
        # time steps
        x_hedge = [j / steps for j in range(steps)]

        # Check if price path is made
        if self.price_path[-1] == 0:
            self.create_price_path()

        # corrected current price for hedge time intervals and all deltas for a given time
        hedge_price = [j for n, j in enumerate(self.price_path) if int(n % (self.steps / steps)) == 0]
        delta_list = [self.hedge(t, s, hedge_setting) for t, s in zip(x_hedge, hedge_price)]

        # New time step and interest for given interval
        dt = self.T / steps
        interest = self.r * dt

        # set iterables
        delta_t = 0
        current_stock_price = 0

        # set loop variables
        previous_delta = delta_t
        bank = self.call_price()

        # loop over the time step and hedge for every time step
        for delta_t, current_stock_price in zip(delta_list, hedge_price):
            cost = (delta_t - previous_delta) * current_stock_price
            bank = bank * math.exp(interest) - cost
            previous_delta = delta_t

        # Calculate the profit when t = T
        profit = bank + (current_stock_price * delta_t) - max([current_stock_price - self.K, 0])

        # Save values for later evaluations
        self.delta_list = delta_list
        self.x_hedge = x_hedge
        self.hedge_price = hedge_price
        
        # return profit made during the hedging
        return profit
    
    def plot_price_path(self, hedge_plot=True):
        '''
        Eneables to plot both price path and delta over time
        '''
        fig, ax1 = plt.subplots()
        
        # Get price time steps
        x_price = [i / self.steps for i in range(self.steps)]
        
        # Plot the price over time on the first axis
        color = 'tab:red'
        ax1.set_xlabel('years')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(x_price, self.price_path, color=color,
                 label="Discritized Black Scholes")
        ax1.tick_params(axis='y', labelcolor=color)

        if hedge_plot:
            # Instantiate a second axes that shares the same x-axis
            ax2 = ax1.twinx()
            color = 'tab:blue'
            # we already handled the x-label with ax1
            ax2.set_ylabel('Delta', color=color)
            # Plot the delta
            ax2.plot(self.x_hedge, self.delta_list, color=color, label='Hedge delta')
            ax2.tick_params(axis='y', labelcolor=color)
           
        # Finalize the plot and show
        plt.title("Stock price and Delta development over time")
        fig.tight_layout()
        plt.show()

    def hedge(self, t, S, hedge_setting='call'):
        '''
        Calculate the delta at a given time
        '''
        # Take d1 from black-scholes
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        
        # Calculate derivitive for call and put
        if hedge_setting.lower() == 'call':
            return st.norm.cdf(d1, 0.0, 1.0)

        elif hedge_setting.lower() == 'put':
            return -st.norm.cdf(d1, 0.0, 1.0)
        else:
            print("Setting not found")
            return None


if __name__ == "__main__":

    for i in range(1):
        B = BlackScholes(1, 100, 99, 0.06, 0.20, 50)
        B.create_price_path()
    #     p = B.create_hedge(200, 'put')
        p = 0
    #     B.plot_price_path()
        p2 = B.create_hedge(50, 'call')
        B.plot_price_path()
        plt.show()
    print(B.put_price(), B.call_price())

    print(([round(p, 1), round(p2, 1)]))

    tree_test = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
                          market="EU", option_type="call", array_out=True)
                          
    price, delta, t_delta, price_tree, option, delta_tree, t_delta_tree = tree_test.determine_price()
    print("Price\n", price)
    print("===============================")
    print("Delta\n", delta)
    print("===============================")
    print("Theoretical Delta\n", t_delta)
    print("===============================")
    print("Price Tree\n", price_tree)
    print("===============================")
    print("Option Tree\n", option)
    print("===============================")
    print("Delta Tree\n", delta_tree)
    print("===============================")
    print("Theoretical Delta Tree\n", t_delta_tree)
    print("===============================")
    # tree1 = BinTreeOption(5, 5 / 12, 50, 0.4, 0.1, 50,
    #                       market="USA", option_type="put", array_out=False)
    # tree2 = BinTreeOption(5, 5 / 12, 50, 0.4, 0.1, 50,
    #                       market="EU", option_type="put", array_out=False)

    # tree3 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
    #                       market="USA", option_type="call", array_out=False)

    # tree4 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
    #                       market="EU", option_type="call", array_out=False)

    # tree5 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
    #                       market="USA", option_type="put", array_out=False)

    # tree6 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
    #                       market="EU", option_type="put", array_out=False)

    # trees = [tree1, tree2, tree3, tree4, tree5, tree6]
    # for i, tree in enumerate(trees):
    #     price, delta = tree.determine_price()
    #     print(f"Price of Tree {i + 1} is", price)
    #     print(f"Delta of Tree {i + 1} is", delta)
    #     print("===============================================")

    # bs_eu = BlackScholes(1, 100, 99, 0.06, 0.2, steps=50)
    # bs_eu.create_price_path()
    # bs_eu.plot_price_path(hedge_setting="Call", hedge_plot=True)
