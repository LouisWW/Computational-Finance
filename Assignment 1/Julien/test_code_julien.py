import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


class BinTreeOption:
    def __init__(
        self, N, T, S0, sigma, r, K,
        market="EU", option_type="call", array_out=False
    ):
        """
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
        assert self.market in [
            "EU", "USA"], "Market not found. Choose EU or USA"
        assert self.option_type in ["call", "put"], "Non-existing option type."

        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)

        # Create price tree and initialize option tree
        self.price_tree = np.zeros((N + 1, N + 1))
        self.create_price_tree()
        self.option = np.zeros((N + 1, N + 1))

    def create_price_tree(self):
        """
        """
        for i in range(self.N + 1):
            for j in range(i + 1):
                self.price_tree[j, i] = self.S0 * \
                    (self.u ** (i - j)) * (self.d ** j)

    def determine_price(self):
        """
        """

        # Calculate vall option price at t=0 (European)
        if self.market == "EU" and self.option_type == "call":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.price_tree[:, self.N] - self.K
            )
            self.recursive_eu_option()

        elif self.market == "EU" and self.option_type == "put":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.K - self.price_tree[:, self.N]
            )
            self.recursive_eu_option()

        elif self.market == "USA" and self.option_type == "call":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.price_tree[:, self.N] - self.K
            )
            self.recursive_usa_call()

        elif self.market == "USA" and self.option_type == "put":
            self.option[:, self.N] = np.maximum(
                np.zeros(self.N + 1), self.K - self.price_tree[:, self.N])
            self.recursive_usa_put()

        if self.array_out:
            return [self.option[0, 0], self.price_tree, self.option]

        return self.option[0, 0]

    def recursive_eu_option(self):
        """
        """
        for i in np.arange(self.N - 1, -1, -1):
            for j in np.arange(0, i + 1):
                self.option[j, i] = (self.discount * (self.p *
                                                      self.option[j, i + 1] + (1 - self.p) *
                                                      self.option[j + 1, i + 1]))

    def recursive_usa_call(self):
        """
        """
        for i in np.arange(self.N - 1, -1, -1):
            for j in np.arange(0, i + 1):
                self.option[j, i] = max([0, self.price_tree[j, i] - self.K,
                                         self.discount *
                                         (self.p * self.option[j, i + 1] +
                                          (1 - self.p) * self.option[j + 1, i + 1])])

    def recursive_usa_put(self):
        """
        """
        for i in np.arange(self.N - 1, -1, -1):
            for j in np.arange(0, i + 1):
                self.option[j, i] = max([0, self.K - self.price_tree[j, i],
                                         self.discount *
                                         (self.p * self.option[j, i + 1] +
                                          (1 - self.p) * self.option[j + 1, i + 1])])

    def reset_tree(self):
        """
        """
        self.price_tree = np.zeros((N + 1, N + 1))
        self.option = np.zeros((N + 1, N + 1))


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

    def call_price(self):
        """
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        call = (self.S0 * st.norm.cdf(d1, 0.0, 1.0) - self.K *
                np.exp(-self.r * self.T) * st.norm.cdf(d2, 0.0, 1.0))

        return call

    def put_price(self):
        """
        """
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        put = ((self.K * np.exp(-self.r * self.T)
                * st.norm.cdf(-d2, 0.0, 1.0)) - self.S0 *
               st.norm.cdf(-d1, 0.0, 1.0))

        return put

    def create_price_path(self):
        """
        """
        for i in range(self.steps):
            self.price_path[i] = self.price
            dS = self.r * self.price * self.dt + self.sigma * \
                self.price * np.random.normal(0, 1) * np.sqrt(self.dt)

            self.price += dS

    def plot_price_path(self, hedge_setting='Call', hedge_plot=True, steps=1):
        fig, ax1 = plt.subplots()
        x_price = [i / self.steps for i in range(self.steps)]

        color = 'tab:red'
        ax1.set_xlabel('years')
        ax1.set_ylabel('Price', color=color)
        ax1.plot(x_price, self.price_path, color=color,
                 label="Discritized Black Scholes")
        ax1.tick_params(axis='y', labelcolor=color)

        if hedge_plot:
            x_hedge = [i / steps for i in range(steps)]
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            # we already handled the x-label with ax1
            ax2.set_ylabel('Delta', color=color)
            h = [self.hedge(t, s, hedge_setting)
                 for t, s in zip(x_hedge, self.price_path)]
            ax2.plot(x_hedge, h, color=color, label='Hedge delta')
            ax2.tick_params(axis='y', labelcolor=color)

        plt.title("Stock price and Delta development over time")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()

    def hedge(self, t, S, hedge_setting='call'):
        hedge_setting = hedge_setting.lower()
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        if hedge_setting == 'call':
            return st.norm.cdf(d1, 0.0, 1.0)

        elif hedge_setting == 'put':
            return -st.norm.cdf(-d1, 0.0, 1.0)
        else:
            print("Setting not found")
            return None


if __name__ == "__main__":

    tree1 = BinTreeOption(5, 5 / 12, 50, 0.4, 0.1, 50,
                          market="USA", option_type="put", array_out=False)
    tree2 = BinTreeOption(5, 5 / 12, 50, 0.4, 0.1, 50,
                          market="EU", option_type="put", array_out=False)

    tree3 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
                          market="USA", option_type="call", array_out=False)

    tree4 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
                          market="EU", option_type="call", array_out=False)

    tree5 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
                          market="USA", option_type="put", array_out=False)

    tree6 = BinTreeOption(50, 1, 100, 0.2, 0.06, 99,
                          market="EU", option_type="put", array_out=False)

    trees = [tree1, tree2, tree3, tree4, tree5, tree6]
    for i, tree in enumerate(trees):
        print(f"Price of Tree {i + 1} is", tree.determine_price())
        print("===============================================")

    bs_eu = BlackScholes(1, 100, 99, 0.06, 0.2, steps=50)
    bs_eu.create_price_path()
    bs_eu.plot_price_path(hedge_setting="Call", hedge_plot=True)
