# Documentation
------------ 

In this Assignment, the PDE is used to price the options. In the main, 3 different functions can be called from main.

The first function is called **test** which return the option price, the different greeks and the visualisation of the grid based on the optimal parameters
  
    s_min=0,s_max=200, ds=1, t_max=1, dt=0.001, S0=100, K=100, r=0.04,sigma=0.3, option='call', fm_type='crank-nicolson'


The second function is called **diff_S0()** which returns the option price calculated with the Crank-Nicolson and FTCS method.
Thereby, the strike price is set to 100,110,120 respectively. The default parameters are 

    s_min=0,s_max=300,ds=1,t_max=1,dt=(0.0001,0.01),S0=(100,110,120),K=110,r=0.04,sigma=0.3,option='call'

where dt=0.001 is used for the FTCS and dt=0.01 is used for the Crank-Nicolson method.


The third function **convergence** returns a 3D plot of the option price for each ds and dt. Thus the aim is to visualise for which grid size the FTCS and Crank-Nicolson methods are unstable. The default parameters are

    s_min=0, s_max=100, t_max=1, S0=50, K=60, r=0.04, sigma=0.2, option='call', fm_type='forward'
