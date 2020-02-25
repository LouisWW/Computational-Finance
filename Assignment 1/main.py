#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""


import helper as helper

'''
exercise 1.1 : Determining the effect of the volatility on the option price
'''
# helper.binomial_tree_1(50, 1, 100, 99, 0.06,'EU',"call",save_plot=True)
# helper.binomial_tree_1(50, 1, 100, 99, 0.06,'EU',"put",save_plot=True)
# helper.binomial_tree_1(50, 1, 100, 99, 0.06,'USA',"call",save_plot=True)
# helper.binomial_tree_1(50, 1, 100, 99, 0.06,'USA',"put",save_plot=True)


'''
exercise 1.2 : Determining the effect of the step used for the binomial tree on the resulting option price

Warning : This function may take a certain time. To boost the performance define NUM_CORE
            
'''
#helper.binomial_tree_2(1, 100, 99, 0.06, 0.2,'EU',"call",save_plot=True,run_time=True)
#helper.binomial_tree_2(1, 100, 99, 0.06, 0.2,'EU',"put",save_plot=True,run_time=True)
#helper.binomial_tree_2(1, 100, 99, 0.06, 0.2,'USA',"call",save_plot=True,run_time=True)
#helper.binomial_tree_2(1, 100, 99, 0.06, 0.2,'USA',"put",save_plot=True,run_time=True)


'''
exercise 1.3 : Determining the effect of the volatilty on the hedging parameter delta
'''
# helper.binomial_tree_3(50,1, 100, 99, 0.06,'EU',"call",save_plot=True)


'''
exercise 1.3 : Plot the stock price based on the Wiener process
'''
# helper.wiener_process(1, 100, 99, 0.06, 0.2, steps=365,save_plot=True)


'''
exercise 2 : Analyse the dynamical hedging daily vs weekly
             and plot the distribution of the profit using different steps
'''
# helper.profit_histogram()
# helper.all_profit_histograms()


'''
exercise 2 : Applying dynamical hedging to AAPL and Shell stock
'''
# helper.real_stock_data()
