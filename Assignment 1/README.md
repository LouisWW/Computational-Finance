# Documentation
------------ 

In this Assignment, the binomial tree is computed and compared to the Black-Scholes solution. 
The influence of the volatility as well as the size of the binomial tree is analysed. 
Finally, the efficiency of dynamical hedging is introduced in order to create a risk-free portfolio.
To compute the option price and the hedging, the following command is necessary where one needs to add the different flags. 
The following command shows the necessary inputs and their default values

    ./main.py --help
    
For example the following command will run the daily hedging vs weekly 
hedging using the default parameters
    
    ./main.py -func 2
