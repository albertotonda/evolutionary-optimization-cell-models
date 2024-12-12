# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:21:26 2024

Script for the new multi-objective problem. This time, we are going to use
inspyred, because it's easier to implement new operators and hybrid genomes.

@author: Alberto
"""
import inspyred
import random

@inspyred.ec.variators.mutator
def specialized_mutation(random : random.Random, candidate : dict, args : dict) :
    """
    This is a specialized mutation, tailored to the specific genome we are using.
    """
    n_variables = args["n_variables"]
    bounder = args['_ec'].bounder # this will be used for the values
    mutation_strength = args["mutation_strength"]
    
    # mean and stdev of the Gaussian are hard-coded, they could be self-adapted
    mean = 0.0
    stdev = 0.1
    
    # get 'mutation_strength', if available
    mutation_strength = args.get('mutation_strength', 0.5) 
    
    # first, let's create a new dictionary
    child = {'arcs' : candidate['arcs'].copy(),
             'values' : candidate['values'].copy()
             }
    
    # then, we need to decide what to do; throwing some random numbers around
    # we first pick whether we are going to modify the arcs or the values
    n_mutations = 0 # this is just used to check that we mutated at least once
    while random.uniform(0, 1) < mutation_strength or n_mutations == 0 :
        
        if random.uniform(0, 1) < 0.5 :
            # modify one of the arcs
            index = random.choice(range(0, n_variables))
            
            if child['arcs'][index] == 0 :
                child['arcs'][index] = 1
            else :
                child['arcs'][index] = 0
                
            n_mutations += 1
            
        else :
            # modify one of the values; but ONLY the values that are actually
            # 'active' (so, the arcs that are equal to 1)
            valid_indexes = [ i for i in range(0, n_variables)
                             if child['arcs'][i] == 1 ]
            # however, the list might be empty! in that case, we do nothing
            if len(valid_indexes) >= 1 :
                index = random.choice(valid_indexes)                
                child['values'][index] = random.gauss(mean, stdev)
                n_mutations += 1
            
    # after potentially several iterations, apply bounder on the values
    # and return the child
    child['values'] = bounder(child['values'], args)
    
    return child

@inspyred.ec.variators.crossover
def specialized_crossover(random : random.Random, parent1 : dict, parent2 : dict, args :dict) :
    """
    This is a specialized crossover operators, taking into account that the individuals
    actually contain two vectors of values (one binary, for the arcs, one float,
    for the values).    
    """
    crossover_rate = args.get("crossover_rate", 0.8)
    n_variables = args["n_variables"]

    # create copies of the parents
    child1 = {k : v.copy() for k, v in parent1.items()}
    child2 = {k : v.copy() for k, v in parent2.items()}
    
    if random.uniform(0, 1) < crossover_rate :    
        # select an index (this is a one-point crossover)
        index = random.choice(range(0, n_variables))
        
        # swap parts
        child1['arcs'][:index] = parent2['arcs'][:index]
        child1['values'][:index] = parent2['values'][:index]
        
        child2['arcs'][index:] = parent1['arcs'][index:]
        child2['values'][index:] = parent1['values'][index:]
    
    # return children as a list
    return [child1, child2]
    
def generator(random : random.Random, args : dict) :
    """
    Generate initial population. Each candidate is a dictionary containing
    two parts: an array of boolean values, representing the arcs in the graph,
    and an array of real values.
    """
    n_variables = args["n_variables"]
    
    individual = {
        'arcs' : [random.choice([0,1]) for _ in range(0, n_variables)], 
        'values' : [random.uniform(0,1) for _ in range(0, n_variables)]
                  }
    
    return individual

@inspyred.ec.evaluators.evaluator
def evaluator(candidate, args) :
    """
    This evaluator is only a placeholder for the moment; since I need a multi-objective
    problem, I will try at the same time to minimize the number of arcs at 1,
    and maximize the sum of all values that have corresponding arcs at 1.
    """
    n_variables = args["n_variables"]
    
    # first fitness function: minimize number of arcs at 1
    fitness_min_arcs = sum(candidate['arcs'])
    
    # second fitness function: maximize sum of values that have arcs at 1
    valid_indexes = [i for i in range(0, n_variables) 
                     if candidate['arcs'][i] == 1]
    fitness_max_values = sum([candidate['values'][i] for i in valid_indexes])
    
    return inspyred.ec.emo.Pareto([fitness_min_arcs, -fitness_max_values])

def observer(population, num_generations, num_evaluations, args) :
    """
    Function called at the end of each generation, for printouts, saving to file,
    and maybe updating some hyperparameters.
    """
    fitness_names = args["fitness_names"]
    mutation_strength = args["mutation_strength"]
    mutation_strength_decay = args["mutation_strength_decay"]
    
    # find the lowest values for each fitness
    best_fitness_values = [ min([i.fitness[f] for i in population]) 
                           for f in range(0, len(fitness_names)) ]
    
    print("Generation: %d, Evaluations: %d, Best fitness values: %s" % 
          (num_generations, num_evaluations, str(best_fitness_values)))
    
    # update hyperparameters
    args["mutation_strength"] = mutation_strength * mutation_strength_decay
    
    return
    
if __name__ == "__main__" :
    
    # hard-coded values
    fitness_names = ["fitness_1", "fitness_2"]
    n_variables = 10
    population_size = 100
    offspring_size = 100
    max_generations = 1000
    
    initial_mutation_strength = 0.9
    mutation_strength_decay = 0.99
    
    random_seed = 42
    
    # set up the evolutionary algorithm
    # first, initialize the pseudo-random number generator
    prng = random.Random()
    prng.seed(random_seed)
    
    # then, set up NSGA2
    nsga2 = inspyred.ec.emo.NSGA2(prng)
    bounder = inspyred.ec.Bounder(lower_bound=0.0, upper_bound=1.0)
    
    nsga2.observer = observer
    nsga2.variator = [specialized_crossover, specialized_mutation]
    nsga2.terminator = inspyred.ec.terminators.generation_termination
    
    final_archive = nsga2.evolve(
        generator = generator,
        evaluator = evaluator,
        pop_size = population_size,
        num_selected = offspring_size,
        maximize = False,
        bounder = bounder,
        max_generations = max_generations,
        
        # all this stuff below will end up in 'args'
        n_variables = n_variables,
        mutation_strength = initial_mutation_strength,
        mutation_strength_decay = mutation_strength_decay,
        fitness_names = fitness_names,
        )
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    fitness_1 = [i.fitness[0] for i in final_archive]
    fitness_2 = [i.fitness[1] for i in final_archive]
    ax.scatter(fitness_1, fitness_2)
    ax.set_xlabel(fitness_names[0])
    ax.set_ylabel(fitness_names[1])
    plt.show()
    
    # find individual with the lowest value of fitness 1
    best_fitness_1 = min(fitness_1)
    best_fitness_1_index = fitness_1.index(best_fitness_1)
    print(final_archive[best_fitness_1_index].candidate)
    
    # actually, let's print everything
    for individual in final_archive :
        print(individual)