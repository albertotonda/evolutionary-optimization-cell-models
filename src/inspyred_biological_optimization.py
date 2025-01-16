# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:21:26 2024

Script for the new multi-objective problem. This time, we are going to use
inspyred, because it's easier to implement new operators and hybrid genomes.

@author: Alberto
"""
import inspyred
import os
import pandas as pd
import random

@inspyred.ec.variators.mutator
def specialized_mutation(random : random.Random, candidate : dict, args : dict) :
    """
    This is a specialized mutation, tailored to the specific genome we are using.
    """
    n_variables_ereg = args["n_variables_ereg"]
    n_variables_ekin = args["n_variables_ekin"]
    bounder_ereg = args["bounder_ereg"]
    bounder_ekin = args["bounder_ekin"]
    mutation_strength = args["mutation_strength"]
    
    # mean and stdev of the Gaussian are hard-coded, they could be self-adapted
    mean = 0.0
    stdev = 0.1
    
    # get 'mutation_strength', if available
    mutation_strength = args.get('mutation_strength', 0.5) 
    
    # first, let's create a new dictionary
    child = {
        'ereg_arcs' : candidate['ereg_arcs'].copy(),
        'ereg_values' : candidate['ereg_values'].copy(),
        'ekin_values' : candidate['ekin_values'].copy()
             }
    
    # then, we need to decide what to do; throwing some random numbers around
    # we first pick whether we are going to modify the arcs or the values
    n_mutations = 0 # this is just used to check that we mutated at least once
    while random.uniform(0, 1) < mutation_strength or n_mutations == 0 :
        
        random_number = random.uniform(0, 1)
        if random_number <= 0.34 :
            # modify one of the arcs
            index = random.choice(range(0, n_variables_ereg))
            
            if child['ereg_arcs'][index] == 0 :
                child['ereg_arcs'][index] = 1
            else :
                child['ereg_arcs'][index] = 0
                
            n_mutations += 1
            
        elif random_number > 0.34 and random_number <= 0.67 :
            # modify one of the values; but ONLY the values that are actually
            # 'active' (so, the arcs that are equal to 1)
            valid_indexes = [ i for i in range(0, n_variables_ereg)
                             if child['ereg_arcs'][i] == 1 ]
            
            # however, the list might be empty! in that case, we do nothing
            if len(valid_indexes) >= 1 :
                index = random.choice(valid_indexes)                
                child['ereg_values'][index] += random.gauss(mean, stdev)
                n_mutations += 1
        
        else :
            # modify one of the fixed ekin values
            index = random.choice(range(0, n_variables_ekin))
            child['ekin_values'][index] += random.gauss(mean, stdev)
            n_mutations += 1
            
    # after potentially several iterations, apply bounder on the values
    # and return the child
    child['ereg_values'] = bounder_ereg(child['ereg_values'], args)
    child['ekin_values'] = bounder_ekin(child['ekin_values'], args)
    
    return child

@inspyred.ec.variators.crossover
def specialized_crossover(random : random.Random, parent1 : dict, parent2 : dict, args :dict) :
    """
    This is a specialized crossover operators, taking into account that the individuals
    actually contain two vectors of values (one binary, for the arcs, one float,
    for the values).    
    """
    crossover_rate = args.get("crossover_rate", 0.8)
    n_variables_ereg = args["n_variables_ereg"]
    n_variables_ekin = args["n_variables_ekin"]

    # create copies of the parents
    child1 = {k : v.copy() for k, v in parent1.items()}
    child2 = {k : v.copy() for k, v in parent2.items()}
    
    if random.uniform(0, 1) < crossover_rate :    
        # select an index (this is a one-point crossover)
        index_ereg = random.choice(range(0, n_variables_ereg))
        
        # swap parts; arcs and corresponding values
        child1['ereg_arcs'][:index_ereg] = parent2['ereg_arcs'][:index_ereg]
        child1['ereg_values'][:index_ereg] = parent2['ereg_values'][:index_ereg]
        
        child2['ereg_arcs'][index_ereg:] = parent1['ereg_arcs'][index_ereg:]
        child2['ereg_values'][index_ereg:] = parent1['ereg_values'][index_ereg:]
        
        # also swap ekin values
        index_ekin = random.choice(range(0, n_variables_ekin))
        child1['ekin_values'][:index_ekin] = parent2['ekin_values'][:index_ekin]
        child2['ekin_values'][index_ekin:] = parent1['ekin_values'][index_ekin:]
    
    # return children as a list
    return [child1, child2]
    
def generator(random : random.Random, args : dict) :
    """
    Generate initial population. Each candidate is a dictionary containing
    two parts: an array of boolean values, representing the arcs in the graph,
    and a corresponding array of real values; then, a real-valude array of different size
    """
    n_variables_ereg = args["n_variables_ereg"]
    n_variables_ekin = args["n_variables_ekin"]
    
    individual = {
        'ereg_arcs' : [random.choice([0,1]) for _ in range(0, n_variables_ereg)], 
        'ereg_values' : [random.uniform(-1,1) for _ in range(0, n_variables_ereg)],
        'ekin_values' : [random.uniform(0,1) for _ in range(0, n_variables_ekin)]
                  }
    
    return individual

@inspyred.ec.evaluators.evaluator
def evaluator(candidate, args) :
    """
    This evaluator is only a placeholder for the moment; since I need a multi-objective
    problem, I will try at the same time to minimize the number of arcs at 1,
    and maximize the sum of all values that have corresponding arcs at 1.
    """
    n_variables_ereg = args["n_variables_ereg"]
    n_variables_ekin = args["n_variables_ekin"]
    
    # first fitness function: minimize number of arcs at 1
    fitness_min_arcs = sum(candidate['ereg_arcs'])
    # and also minimize the sum of ekin_values
    fitness_min_arcs += sum(candidate['ekin_values'])
    
    # second fitness function: minimize sum of values that have arcs at 1,
    # this should lead to a negative number, as ereg_values can range in (-1,1)
    valid_indexes = [i for i in range(0, n_variables_ereg) 
                     if candidate['ereg_arcs'][i] == 1]
    fitness_min_values = sum([candidate['ereg_values'][i] for i in valid_indexes])
    
    # let's also add ekin_values to the mix; these values range in (0,1), so we
    # add them up and subtract them from the fitness_min_values, creating a
    # pressure to maximize the sum
    fitness_min_values += -sum(candidate['ekin_values'])
    
    return inspyred.ec.emo.Pareto([fitness_min_arcs, fitness_min_values])

def observer(population, num_generations, num_evaluations, args) :
    """
    Function called at the end of each generation, for printouts, saving to file,
    and maybe updating some hyperparameters.
    """
    fitness_names = args["fitness_names"]
    mutation_strength = args["mutation_strength"]
    mutation_strength_decay = args["mutation_strength_decay"]
    output_folder = args["output_folder"]
    
    # find the lowest values for each fitness
    best_fitness_values = [ min([i.fitness[f] for i in population]) 
                           for f in range(0, len(fitness_names)) ]
    
    print("Generation: %d, Evaluations: %d, Best fitness values: %s" % 
          (num_generations, num_evaluations, str(best_fitness_values)))
    
    # save population
    if num_generations % 10 == 0 :
        
        file_name = os.path.join(output_folder, "population-generation-%d.csv" % num_generations)
        
        # we are going to create a dictionary, then convert it to a DatFrame
        # and save it as a CSV
        population_dictionary = {}
        population_dictionary['generation'] = [num_generations] * len(population)
        for i, fitness_name in enumerate(fitness_names) :
            population_dictionary[fitness_name] = [individual.fitness[i] for
                                                   individual in population]
        
        for i in range(0, len(population[0].candidate['ereg_arcs'])) :
            population_dictionary['ereg_arcs_%d' % i] = [individual.candidate['ereg_arcs'][i] for
                                                         individual in population]
        for i in range(0, len(population[0].candidate['ereg_values'])) :
            population_dictionary['ereg_values_%d' % i] = [individual.candidate['ereg_values'][i] for
                                                           individual in population]
        for i in range(0, len(population[0].candidate['ekin_values'])) :
            population_dictionary['ekin_values_%d' % i] = [individual.candidate['ekin_values'][i] for
                                                           individual in population]
        
        df = pd.DataFrame.from_dict(population_dictionary)
        df.to_csv(file_name, index=False)
    
    # update hyperparameters
    args["mutation_strength"] = mutation_strength * mutation_strength_decay
    
    return
    
if __name__ == "__main__" :
    
    # TODO modify the code so that ereg is numbers in (0,1) with booleans
    # associated
    
    # hard-coded values
    fitness_names = ["fitness_1", "fitness_2"]
    output_folder = "../local/" + os.path.basename(__file__)[:-3]
    n_variables_ereg = 10 # ekin and ereg are the two parts of the elasticity matrix
    n_variables_ekin = 5
    population_size = 100
    offspring_size = 100
    max_generations = 200
    
    initial_mutation_strength = 0.9
    mutation_strength_decay = 0.99
    
    random_seed = 42
    
    # set up the evolutionary algorithm
    # first, initialize the pseudo-random number generator
    prng = random.Random()
    prng.seed(random_seed)
    
    # then, set up NSGA2
    nsga2 = inspyred.ec.emo.NSGA2(prng)
    
    # specific bounders for each separate set of values
    bounder_ereg = inspyred.ec.Bounder(lower_bound=-1.0, upper_bound=1.0)
    bounder_ekin = inspyred.ec.Bounder(lower_bound=0.0, upper_bound=1.0)
    
    nsga2.observer = observer
    nsga2.variator = [specialized_crossover, specialized_mutation]
    nsga2.terminator = inspyred.ec.terminators.generation_termination
    
    # create the output folder
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
    
    final_archive = nsga2.evolve(
        generator = generator,
        evaluator = evaluator,
        pop_size = population_size,
        num_selected = offspring_size,
        maximize = False,
        max_generations = max_generations,
        
        # all this stuff below will end up in 'args'
        n_variables_ereg = n_variables_ereg,
        n_variables_ekin = n_variables_ekin,
        bounder_ereg = bounder_ereg,
        bounder_ekin = bounder_ekin,
        mutation_strength = initial_mutation_strength,
        mutation_strength_decay = mutation_strength_decay,
        fitness_names = fitness_names,
        output_folder = output_folder,
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