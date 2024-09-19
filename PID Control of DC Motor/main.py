# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:50:46 2024

@author: mohamed
"""

import numpy as np
from DC_Model import DCmotorPID
from Metas import SA, EP, PSO, GA


# Creating DC-PID Controller object
DC1 = DCmotorPID(0.5, 3.0, 0.2, 0.2, 1, 0.7, 0.8) 



# define lower boundries, upper boundries and Number of iterations
x_max = np.array([3]*3)
x_min = np.array([0.0001]*3)
Iterations = 100



# Running the Algorithm ( you can choose any one of them to run)
EP_res = EP(DC1.evaluate,3,x_min,x_max,30,Iterations,0.1,5)

GA_res = GA(DC1.evaluate,3,x_min,x_max,generations=Iterations)

PSO_res = PSO(DC1.evaluate,3,x_min,x_max,30,Iterations,weight_decay=0.99
              ,c1=2,c2=4,Sgmnt=10,Weight=1)

SA_res = SA(DC1.evaluate, x_min, x_max, Iterations, cycles=100, decay=0.9)







# Visualizing the optimization process
best_function_values = [EP_res[0][i] for i in range(200)]    
plt.figure(figsize=(8, 6))
plt.plot(best_function_values)
plt.title('Optimization Process over iterations')
plt.xlabel('Iterations')
plt.ylabel('Function Value')


