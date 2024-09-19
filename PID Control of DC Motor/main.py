# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:50:46 2024

@author: mohamed
"""

import numpy as np
from DC_Model import DCmotorPID
from Metas import SA, EP, PSO, GA

DC1 = DCmotorPID(0.5, 3.0, 0.2, 0.2, 1, 0.7, 0.8) 

#PIDfactors = np.array([0.22323,0.67517,0.65175])



###################################################
###################################################
###################################################
###################################################


x_max = np.array([3]*3)
x_min = np.array([0.0001]*3)

EP_res = EP(DC1.evaluate,3,x_min,x_max,30,200,0.1,5)

GA_res = GA(DC1.evaluate,3,x_min,x_max)

PSO_res = PSO(DC1.evaluate,3,x_min,x_max,30,200,weight_decay=0.99
              ,c1=2,c2=4,Sgmnt=10,Weight=1)

SA_res = SA(DC1.evaluate, x_min, x_max, iterations=100, cycles=100, decay=0.9)


###################################################
###################################################
###################################################
###################################################













'''
nx = len(x_max)
nP = 30
nG = 200

alpha = 0.99
constant1 = 2
constant2 = 4
sgmnt = 7
weight = 1

particles = np.zeros([nP,nx])
velocities = np.zeros([nP,nx])
results =  np.zeros([nP,1])

max_velocities = (x_max - x_min) / sgmnt

#### Initialization of first generation
#### particles and their velocities 
for i in range(nP):
    
    particles[i] = x_min*np.ones([1,nx]) + np.random.rand(3)*(x_max-x_min)
    
    velocities[i] = -max_velocities*np.ones([1,nx]) + np.random.rand(3)*(2*max_velocities)
 
 
    
#### Evaluate First generation 
for i in range(nP):
    results[i] = DC1.evaluate(particles[i])
    

#### search for individual best 
best_particles_local = particles
best_function_local = results

#### search for global best 
best_particles_global = [particles[0]]
best_function_global = [results[0]]

for i in range(1,nP):
    
    if best_function_local[i] < best_function_global[0]:
        best_particles_global[0] = best_particles_local[i]
        best_function_global[0] = best_function_local[i]


for j in range(1,nG):
    
    #### weight update 
    weight = alpha*weight 
    
    #### velocity update
    for i in range(nP):
        
        for k in range(nx):
            
             a = weight*velocities[i,k]
             b = constant1*np.random.rand()*(best_particles_local[i,k] - particles[i,k])
             c = constant2*np.random.rand()*(best_particles_global[-1][k] - particles[i,k])
             velocities[i,k] = a + b + c
    
    
    
    #### check velocity limits 
    for i in range(nP):
        
        for k in range(nx):
            
            if velocities[i,k] > max_velocities[k]:
               velocities[i,k] = max_velocities[k]
            
            if velocities[i,k] < (-max_velocities[k]):
               velocities[i,k] = (-max_velocities[k])
    
    
    
    #### update particles (candidate solutions)
    for i in range(nP):
        
        particles[i] = velocities[i] + particles[i]
    
    
    
    #### check particles limits 
    for i in range(nP):
        
        for k in range(nx):
            
            if particles[i,k] > x_max[k]:
               particles[i,k] = x_max[k]
            
            if particles[i,k] < x_min[k]:
               particles[i,k] = x_min[k]
    
    
    #### Evaluate current generation 
    for i in range(nP):
        results[i] = DC1.evaluate(particles[i])
    
    
    #### search for individual best
    for i in range(nP):
        
        if results[i] < best_function_local[i]:
            best_function_local[i] = results[i]
            best_particles_local[i] = particles[i]
            
        
    #### search for global best 
    best_particles_global.append(best_particles_global[-1])
    best_function_global.append(best_function_local[-1])

    for i in range(1,nP):
    
        if best_function_local[i] < best_function_global[-1]:
            best_particles_global[-1] = best_particles_local[i]
            best_function_global[-1] = best_function_local[i]

'''





















'''
nx = len(x_max)
nP = 30
nG = 200
Beta = 0.1
solutions = np.zeros([2*nP,nx])
results =  np.zeros([2*nP,1])

#### Initialization of first generation
for i in range(nP):
    solutions[i] = x_min*np.ones([1,nx]) + np.random.rand(3)*(x_max-x_min)
    
#### Evaluate First generation 
for i in range(nP):
    results[i] = DC1.evaluate(solutions[i])
    
#### Best Candidate
sol_best = [solutions[0]]
res_best = [results[0]]
res_min = results[0]
res_max = results[0]

for i in range(nP):
    
    if results[i] < res_best[0] : 
        res_best[0] = results[i]
        sol_best[0] = solutions[i]
    
    if results[i] > res_max : 
        res_max = results[i]
    
res_min = res_best[0]

#### Evolution
#### iterate over generations 

for i in range(1,nG):
  
    #### Mutation
    #### Computing the second half of population 
    for j in range(nP):
        
        for k in range(nx):
            
            sigma = abs(Beta * (results[j]/res_max) * (x_max[k] - x_min[k]) )
            solutions[nP+j,k] = solutions[j,k] + sigma*np.random.rand()
    
    #### bounds violation treatment 
    for j in range(2*nP):
        
        for k in range(nx):
            
            if solutions[j,k] > x_max[k]:
                solutions[j,k] = x_max[k]
            
            if solutions[j,k] < x_min[k]:
                solutions[j,k] = x_min[k]
        
    
    #### Evaluate fitness of current generation 
    for ii in range(2*nP):
        results[ii] = DC1.evaluate(solutions[ii])
    
    #### Best Candidate
    sol_best.append(sol_best[-1])
    res_best.append(res_best[-1])
    res_min = results[0]
    res_max = results[0]

    for jj in range(2*nP):
    
        if results[jj] < res_best[-1] : 
            res_best[-1] = results[jj]
            sol_best[-1] = solutions[jj]
    
        if results[jj] > res_max : 
            res_max = results[jj]
        
        if results[jj] < res_min : 
            res_min = results[jj]
    
    #### Tournament 
    q = 2*nP - 5
    weights = np.zeros([2*nP,1])
    
    for j in range(2*nP):
        
        #rand_k = np.random.permutation(2*nP)
        
        for m in range(q):    
            rand_k = np.random.permutation(2*nP)
            d = rand_k[0]
            
            if d == j:
                d = rand_k[1]
                #continue
            
            if np.random.rand() > ( results[j] / (results[j]+results[d]) ) :
                weights[j] += 1
            
    #### Ranking 
    sorted_indices = np.flip(np.argsort(weights.T))
    sorted_indices = np.reshape(sorted_indices,(2*nP,))
    weights = weights[sorted_indices]
    solutions = solutions[sorted_indices]
    results = results[sorted_indices]
    
    
    #### selection
    # it is already made by ranking the weights of the candidate solutions
    # in descending order (minimization) , the first np are the selected individuals 
    '''
 
    
'''
nx = len(x_max)
nP = 30
nG = 200
Beta = 0.1
current_solutions = np.zeros([nP,nx])
results =  np.zeros([nP,1])
new_solutions = np.zeros([nP,nx])
new_results = np.zeros([nP,1])

#### Initialization of first generation
for i in range(nP):
    current_solutions[i] = x_min*np.ones([1,nx]) + np.random.rand(3)*(x_max-x_min)
    
#### Evaluate First generation 
for i in range(nP):
    results[i] = DC1.evaluate(current_solutions[i])
    
#### Best Candidate
sol_best = [current_solutions[0]]
res_best = [results[0]]
res_min = results[0]
res_max = results[0]

for i in range(nP):
    
    if results[i] < res_best[0] : 
        res_best[0] = results[i]
        sol_best[0] = current_solutions[i]
    
    if results[i] > res_max : 
        res_max = results[i]
    
res_min = res_best[0]

#### Evolution
#### iterate over generations 

for i in range(1,nG):
  
    #### Selection
    
    #### Crossover
    
    #### Mutation 
    
    #### bounds violation treatment 
    for j in range(2*nP):
        
        for k in range(nx):
            
            if solutions[j,k] > x_max[k]:
                solutions[j,k] = x_max[k]
            
            if solutions[j,k] < x_min[k]:
                solutions[j,k] = x_min[k]
        
    
    #### Evaluate fitness of current generation 
    for ii in range(2*nP):
        results[ii] = DC1.evaluate(solutions[ii])
    
    #### Best Candidate
    sol_best.append(sol_best[-1])
    res_best.append(res_best[-1])
    res_min = results[0]
    res_max = results[0]

    for jj in range(2*nP):
    
        if results[jj] < res_best[-1] : 
            res_best[-1] = results[jj]
            sol_best[-1] = solutions[jj]
    
        if results[jj] > res_max : 
            res_max = results[jj]
        
        if results[jj] < res_min : 
            res_min = results[jj]
    
    #### Tournament 
    q = 2*nP - 5
    weights = np.zeros([2*nP,1])
    
    for j in range(2*nP):
        
        #rand_k = np.random.permutation(2*nP)
        
        for m in range(q):    
            rand_k = np.random.permutation(2*nP)
            d = rand_k[0]
            
            if d == j:
                d = rand_k[1]
                #continue
            
            if np.random.rand() > ( results[j] / (results[j]+results[d]) ) :
                weights[j] += 1
            
    #### Ranking 
    sorted_indices = np.flip(np.argsort(weights.T))
    sorted_indices = np.reshape(sorted_indices,(2*nP,))
    weights = weights[sorted_indices]
    solutions = solutions[sorted_indices]
    results = results[sorted_indices]
    '''