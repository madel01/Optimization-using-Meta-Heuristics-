# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:15:59 2024

@author: mohamed
"""

import numpy as np
import random 

def SA(obj,x_min_bound,
       x_max_bound, 
       initial_temp=384, 
       iterations=3, 
       cycles=3, 
       step=3, 
       decay=0.5, 
       tempStop=20):

    temperature = initial_temp
    n = iterations
    cycle_number = cycles
    step_size = step


    x = x_min_bound + (np.random.rand(3))*(x_max_bound-x_min_bound)
    func = obj(x)
    points = [x]
    function_values = [func]
    best_point = x
    best_function_value = func

    while(temperature >= tempStop):
    
        i = 1
    
        while (i < n) :
        
              #points.append(x)
        
              temp_rand = np.random.rand(3)
              x_next = (x - step_size) + (temp_rand * (2*step_size))
              
              for k in range(3):
            
                if x_next[k] > x_max_bound[k]:
                   x_next[k] = x_max_bound[k]
            
                if x_next[k] < x_min_bound[k]:
                   x_next[k] = x_min_bound[k]
        
              func1 = obj(x_next)
        
              delta = func1 - func
        
              if delta <= 0 or (random.random() < np.exp( -delta/temperature ) ):
           
                 x = x_next 
                 func = func1
                 
                 if func < best_function_value : 
                     best_function_value = func
                     best_point = x
                     
                 
                 points.append(x)
                 function_values.append(func)
        
              i = i + 1
    
        cycle_number = cycle_number + 1
        temperature = temperature * decay
    
    
    return best_function_value,best_point,points,function_values








def EP(obj,parameters,
       x_min_bound,
       x_max_bound,
       individuals=30,
       generations=200,
       beta=0.01,
       tournament_const=2):
    
    x_min = x_min_bound
    x_max = x_max_bound

    nx = parameters
    nP = individuals
    nG = generations
    Beta = beta
    solutions = np.zeros([2*nP,nx])
    results =  np.zeros([2*nP,1])

    #### Initialization of first generation
    for i in range(nP):
        solutions[i] = x_min*np.ones([1,nx]) + np.random.rand(3)*(x_max-x_min)
    
    #### Evaluate First generation 
    for i in range(nP):
        results[i] = obj(solutions[i])
    
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
            results[ii] = obj(solutions[ii])
    
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
        q = 2*nP - tournament_const
        weights = np.zeros([2*nP,1])
    
        for j in range(2*nP):
        
            for m in range(q):
                
                rand_k = np.random.permutation(2*nP)
            
                d = rand_k[0]
            
                if d == j:
                   d = rand_k[1]
            
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
    
    return res_best,sol_best
    
    
    
    
    

def PSO(obj,parameters,
       x_min_bound,
       x_max_bound,
       individuals=30,
       generations=200,
       weight_decay = 0.99,
       c1 = 2,
       c2 = 4,
       Sgmnt = 7,
       Weight = 1):
    
    x_max = np.array([3]*3)
    x_min = np.array([0.0001]*3)

    nx = parameters
    nP = individuals
    nG = generations

    alpha = weight_decay
    constant1 = c1
    constant2 = c2
    sgmnt = Sgmnt
    weight = Weight

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
        results[i] = obj(particles[i])
    

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
            results[i] = obj(particles[i])
    
    
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
    
    return best_function_global,best_particles_global
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def roulette_wheel_selection(population, fitness):
    # Normalize the fitness values to obtain selection probabilities
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    
    # Calculate cumulative probabilities
    cumulative_probabilities = np.cumsum(probabilities)
    
    # Spin the wheel (random number between 0 and 1)
    rand_1 = np.random.rand()
    rand_2 = np.random.rand()      
    parents = [0]*2
    
    # Select the individual based on where the random number falls in the cumulative probabilities
    for i, cumulative_probability in enumerate(cumulative_probabilities):
        if rand_1 < cumulative_probability:
            parents[0] = population[i]
            break
    
    for i, cumulative_probability in enumerate(cumulative_probabilities):
        if rand_2 < cumulative_probability:
            parents[1] = population[i]
            if np.array_equal(parents[0], parents[1]):
                
                if np.array_equal(parents[1], population[-1]):
                    
                   parents[1] = population[-2]
                
                else:
                
                   parents[1] = population[i+1]
                    
            break
    
    return parents





def Flat_crossover(parentss):
    
    offspring = [0]*2
    
    parent_min = np.minimum(parentss[0],parentss[1])
    parent_max = np.maximum(parentss[0],parentss[1])
    distance = abs(parent_max - parent_min)
                
    offspring[0] = parent_min + np.random.rand(3)*distance
    offspring[1] = parent_min + np.random.rand(3)*distance
    
    return offspring
    

def GA(obj,n_parameters,
       lower_bound,
       upper_bound,
       individuals=30,
       generations=200,
       crossover_prop = 0.9,
       mutation_prop = 0.1):
    
    x_min = lower_bound
    x_max = upper_bound
    
    #x_max = np.array([3]*3)
    #x_min = np.array([0.0001]*3)

    nx = n_parameters
    nP = individuals
    nG = generations
    Pc = crossover_prop
    Pm = mutation_prop
    
    #nx = 3
    #nP = 30
    #nG = 100
    #Pc = 0.9
    #Pm = 0.1
    
    
    
    solutions = np.zeros([nP,nx])
    results =  np.zeros([nP,1])

    #### Initialization of first generation
    for i in range(nP):
        solutions[i] = x_min*np.ones([1,nx]) + np.random.rand(3)*(x_max-x_min)
    
    #### Evaluate First generation 
    for i in range(nP):
        results[i] = obj(solutions[i])
    
    #### Best Candidate
    sol_best = [solutions[0]]
    res_best = [results[0]]
    

    for i in range(nP):
    
        if results[i] < res_best[0] : 
           res_best[0] = results[i]
           sol_best[0] = solutions[i]

    








    #### Evolution
    #### iterate over generations 

    for i in range(1,nG):
  
        
  
        new_solutions = np.zeros([nP,nx])
        fitness = 1 / (results + 1e-6)
        
        for j in range(0,nP,2):
            
            #### Selection
            parents = roulette_wheel_selection(solutions, fitness)
            
            
            #### Cross over
            rand_num = np.random.rand()
            offspring = [0]*2

            if rand_num <= Pc:
                offspring = Flat_crossover(parents)
                
            else:
                offspring = parents
        
        
            #### Mutation
            rand_nums_1 = [np.random.rand(),np.random.rand(),np.random.rand()]
            rand_nums_2 = [np.random.rand(),np.random.rand(),np.random.rand()]
        
            for ii in range(len(rand_nums_1)):
                if rand_nums_1[ii] <= Pm:
                  offspring[0][ii] = x_min[ii] + np.random.rand()*(x_max[ii]-x_min[ii])
                   
        
            for ii in range(len(rand_nums_2)):
                if rand_nums_2[ii] <= Pm:
                  offspring[1][ii] = x_min[ii] + np.random.rand()*(x_max[ii]-x_min[ii])
                   
            
            #### Adding two offsprings to the new population
            new_solutions[j] = offspring[0]
            new_solutions[j+1] = offspring[1]
    
    
    
    
    
    
    
    
    
        #solutions = new_solutions
        #### bounds violation treatment 
        for j in range(nP):
        
            for k in range(nx):
            
                if new_solutions[j,k] > x_max[k]:
                   new_solutions[j,k] = x_max[k]
            
                if new_solutions[j,k] < x_min[k]:
                   new_solutions[j,k] = x_min[k]
        
        
    
        #### Evaluate fitness of current generation
        new_results = np.zeros([nP,1])
        for ii in range(nP):
            new_results[ii] = obj(new_solutions[ii])
    
    
        #### Best Candidate
        sol_best.append(sol_best[-1])
        res_best.append(res_best[-1])
        

        for jj in range(nP):
    
            if new_results[jj] < res_best[-1] : 
               res_best[-1] = new_results[jj]
               sol_best[-1] = new_solutions[jj]
    
    
        solutions = new_solutions
        results = new_results
        
        
        
    return res_best,sol_best
    
    
    
    