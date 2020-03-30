# Documentation
------------ 

In this Assignment, the option price is computed using Monte carlo method. The greek delta 
is computed using the the bump and revalue method. One can choose to either use
fixed seeds or random seed.

To compute the option price and the hedging, the following command is necessary where one needs to add the different 
flags. The following command show the necessary inputs and their default values

    ./main.py --help
    
For example the following command will run the daily hedging vs weekly 
hedging using the default parameters
    
    ./main.py -func  'bump_and_revalue'
  
which return 3 arrays (Monte Carlo result,Black-Scholes solution,Relative error)
 with the column being the different epsilon and the row the number of samples


