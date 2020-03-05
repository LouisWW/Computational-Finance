#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Friday Feb 20 2020
This code was implemented by
Louis Weyland, Floris Fok and Julien Fer
"""

import argparse
import os
import helper as helper



parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Process the binomial Tree and Black Scholes equation given the spot price S0, the strike price K, \
the volatility sigma and the interest rate. \n \
1.1 Determining the effect of the volatility on the option price \n \
1.2 Determining the effect of the step used for the binomial tree on the resulting option price \n \
1.3 Determining the effect of the volatilty on the hedging parameter delta \n \
1.4 Plot the stock price based on the Wiener process \n \
2   Plot the daily hedging vs weekly \n \
3   Plot real data ' )

parser.add_argument("-func",type = float, default=0.0, help='Defines which function to execute')
parser.add_argument('-N', type=int,default=50, help='Size of the binomial tree (default : 50)')
parser.add_argument('-T', type=int,default=1, help='Time to maturity in years (default : 1)')
parser.add_argument('-S', type=int,default=100, help='Stoke price at the moment (default : 100)')
parser.add_argument('-K', type=int,default=99, help='Strike price at the moment (default : 99)')
parser.add_argument('-steps', type=int,default=365, help='Number of updates of the stock market over the year')
parser.add_argument('-r', type=float,default=0.06, help='interest rate r (default : 0.06)')
parser.add_argument('-s', type=float, default=0.2, help='volatility s (default : 0.2)')
parser.add_argument('-option_type', type=str,default='call', help='option type EU or USA(default : EU)')
parser.add_argument('-market', type=str,default='EU', help='option type EU or USA(default : EU)')
parser.add_argument('-save_plot',default=False, help='return the plots (default : False)')
parser.add_argument('-run_time',default=False, help='return the plots (default : False)')
parser=parser.parse_args()


if not os.path.exists('/figures') and parser.save_plot==True:
    os.makedirs('/figures')



if parser.func==0 :
    print("\n\n\n !!! You need to define a funciton !!!  \n\n\n")
    raise AssertionError()


'''
exercise 1.1 : Determining the effect of the volatility on the option price
'''

if parser.func == 1.1:
    helper.binomial_tree_1(parser.N, parser.T, parser.S, parser.K, parser.r, parser.market,parser.option_type, \
                           parser.save_plot)

'''
exercise 1.2 : Determining the effect of the step used for the binomial tree on the resulting option price

Warning : This function may take a certain time. To boost the performance define NUM_CORE
            
'''
if parser.func == 1.2:
    helper.binomial_tree_2(parser.T, parser.S, parser.K, parser.r, parser.s, parser.market,parser.option_type,\
                           parser.save_plot,parser.run_time)

       # 1, 100, 99, 0.06, 0.2,'EU',"call",save_plot=True,run_time=True)

'''
exercise 1.3 : Determining the effect of the volatilty on the hedging parameter delta
'''

if parser.func == 1.3:
 helper.binomial_tree_3(parser.N,parser.T, parser.S, parser.K, parser.r, parser.market,parser.option_type,\
                           parser.save_plot)




'''
exercise 1.4 : Plot the stock price based on the Wiener process
'''
if parser.func == 1.4:
 helper.wiener_process(parser.N,parser.T, parser.S, parser.r, parser.s,parser.steps,parser.save_plot)


'''
exercise 2 : Analyse the dynamical hedging daily vs weekly
             and plot the distribution of the profit using different steps
'''

if parser.func == 2:
    helper.profit_histogram()
    helper.all_profit_histograms()


'''
exercise 2 : Applying dynamical hedging to AAPL and Shell stock
'''
if parser.func == 3:
    helper.real_stock_data()
