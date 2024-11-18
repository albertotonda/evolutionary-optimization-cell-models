# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:31:56 2024

Using pymoo to optimize Arthur's model.

@author: Alberto
"""

# imports
import copy
import datetime
import multiprocessing
import numpy as np
import os
import pandas as pd
import sys

from threading import Lock

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# there is a super-annoying FutureWarning appearing everytime the evaluation
# function is called, so this block of code here just mutes it
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import from local module
from model.cell_model import MODEL

# another import from a local script
from multi_thread_utils import ThreadPool

# specific problem class which inherits from pymoo's generic problem class
class CellProblem(Problem) :
    
    cell_model_instance = None
    n_proc = 10
    parallel_evaluation = False
    
    def __init__(self, n_variables, n_objectives, cell_model_instance, xl=0.0, xu=1.0, n_proc=10, parallel_evaluation=False) :
        # xl and xu are the lower and upper bounds for each variable
        super().__init__(n_var=n_variables, n_obj=n_objectives, xl=0.0, xu=1.0)
        # store the instance of the cell model, to be used later to compute
        # the fitness values
        self.cell_model_instance = cell_model_instance
        # this is potentially used for multiprocessing
        self.parallel_evaluation = parallel_evaluation
        self.n_proc = n_proc
        
        
    def _evaluate(self, x, out, *args, **kwargs) :
        # this evaluation function starts from the assumption that 'x' is actually
        # an array containing all individuals; so we can shape the fitness values
        # numpy array accordingly
        fitness_values = np.zeros((x.shape[0], 1, self.n_obj))
        
        if self.parallel_evaluation :
            # now, it would be great to start a multi-process or multi-thread
            # evaluation; but the bottleneck here is the model instance ; we would
            # need to create independent copies of the model instance, so that each
            # process can run things independently. Maybe we can do it directly
            # inside a specialize multi-processing function
            thread_pool = ThreadPool(self.n_proc)
            thread_lock = Lock()
            
            # create list of arguments for threads; a hidden argument 'thread_id'
            # is added during ThreadPool.map(), so the function called in .map()
            # has to accept a last argument (it's just an integer)
            arguments = [ (x[i], self.cell_model_instance, i, fitness_values, thread_lock) 
                         for i in range(0, x.shape[0]) ]
            # queue function and arguments for the thread pool
            thread_pool.map(multiprocessing_evaluator, arguments)

            # wait the completion of all threads
            thread_pool.wait_completion()
        
        else :
            # run the batch evaluation
            for i in range(0, x.shape[0]) :
                # fitness values are obtained by mapping the vector the cell model's
                # internal elasticity matrix, and then calling a method from the instance
                self.cell_model_instance.elasticity.s.change_from_vector(x[i])
                x_fitness_values = self.cell_model_instance.MOO.list_fitness()
                # store the fitness values, converted to a numpy array
                fitness_values[i,0,:] = np.array(x_fitness_values)
                
        # place the appropriate result in the 'out' dictionary
        out["F"] = fitness_values
        
        return
    
def multiprocessing_evaluator(individual, model, index, fitness_values, lock, thread_id) :
    """
    Fitness function invoked during the multi-processing step; 'thread_id' is
    added by the ThreadPool class, it is supposed to be used for debugging
    """
    # copy the model
    local_model = copy.deepcopy(model)
    # apply modifications and get fitness values
    local_model.elasticity.s.change_from_vector(individual)
    x_fitness_values = local_model.MOO.list_fitness()
    
    # HORRIBLE HACK HERE, BUT IT'S JUST TO SEE IF IT WORKS
    # objectives are rescaled, we know the theoretical maximum of each
    #x_fitness_values[0] /= 7849979.802459471
    #x_fitness_values[1] /= 18.175966975371875
    x_fitness_values[0] /= 1e6
    x_fitness_values[1] /= 100
    
    lock.acquire()
    #print(individual, model, index, fitness_values, lock)
    fitness_values[index,0,:] = np.array(x_fitness_values)
    lock.release()
    
    return
    

# this class here inherits from pymoo.core.callback.Callback, and it is something
# that will be invoked at the end of each iteration of the evolutionary algorithm
class SavePopulationCallback(Callback) :
    
    folder = ""
    population_file_name = ""
    
    # class constructor
    def __init__(self, folder, population_file_name, overwrite_file=False) :
        super().__init__()
        self.folder = folder
        self.population_file_name = population_file_name
        
    # this method is called at every iteration of the algorithm
    def notify(self, algorithm) :
        
        # get the current generation and other information
        generation = algorithm.n_gen
        X = algorithm.pop.get("X")
        F = algorithm.pop.get("F")
        
        results_dictionary = dict()
        results_dictionary["generation"] = [generation] * X.shape[0]
        for i in range(0, F.shape[1]) :
            results_dictionary["fitness_%d" % (i+1)] = F[:,i]
        for i in range(0, X.shape[1]) :
            results_dictionary["variable_%d" % i] = X[:,i]
        df_results = pd.DataFrame.from_dict(results_dictionary)
        df_results.to_csv(os.path.join(self.folder, self.population_file_name + "-%d.csv" % generation), index=False)
        

if __name__ == "__main__" :
    
    # hard-coded values
    population_size = 100
    offspring_size = population_size
    max_generations = 10000
    seed_initial_population_with_prior = False
    
    random_seed = 42
    results_folder = "../local" # 'local' is not under version control (git)
    population_file_name = "%d-population-generation" % random_seed
    
    # generate the folder; the name will be different for every run, as it is
    # initialized with the current time
    output_folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-cell-model-optimization"
    output_folder = os.path.join(results_folder, output_folder)
    
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
        
    # TODO initialize logging
    
    # instantiate model
    model = MODEL()
    # initialize model with default internal values for elasticity matrix
    # Small model of  4 chimical species and  3 reactions =>    6 elasticities but only   4 to evaluate
    #model.MOO.build_model(seed=random_seed)
    # Big   model of 64 chimical species and 57 reactions => 2850 elasticities but only 234 to evaluate
    model.MOO.build_model(source_file="../data/SBtab/E Coli Core/model.tsv", seed=random_seed)
    
    # get the information that we need
    n_variables = model.MOO.vectors["shape"]
    n_objectives = len(model.MOO.list_fitness())
    
    # this is just a debug printout to check that everything is in order
    print(model.MOO.vectors)
    example_individual = model.MOO.vectors["mu"]
    model.elasticity.s.change_from_vector(example_individual)
    #half_saturated_individual = model.elasticity.s.half_satured()
    print(model.MOO.list_fitness())
    #print(model.elasticity.s.df.values.shape)
    
    # and now, we begin for real
    print("Setting up the evolutionary algorithm...")
    
    # let's start with instantiating the problem class
    cell_problem = CellProblem(n_variables, n_objectives, model, n_proc=64, parallel_evaluation=True)
    
    # also, let's create a random initial population, possibly replacing some of the
    # random individuals with specific individuals that are known
    np.random.seed(random_seed)
    initial_population = np.random.random_sample(size=(population_size, n_variables))
    if seed_initial_population_with_prior :
        initial_population[0] = model.MOO.vectors["mu"]
    
    # then, let's set up the algorithm
    algorithm = NSGA2(pop_size=population_size, sampling=initial_population)
    
    # and a callback function
    callback = SavePopulationCallback(output_folder, population_file_name)
    
    # start the run
    print("Starting the evolutionary run, population_size=%d, max_generations=%d" %
          (population_size, max_generations))
    result = minimize(  
                cell_problem,
                algorithm,
                ('n_gen', max_generations),
                callback=callback,
                seed=random_seed,
                verbose=True
                        )
    
    # do something with the result
    results_dictionary = dict()
    results_dictionary["generation"] = [max_generations] * result.X.shape[0]
    for i in range(0, n_objectives) :
        results_dictionary["fitness_%d" % (i+1)] = result.F[:,i]
    for i in range(0, n_variables) :
        results_dictionary["variable_%d" % i] = result.X[:,i]
    df_results = pd.DataFrame.from_dict(results_dictionary)
    df_results.to_csv(os.path.join(output_folder, "result.csv"), index=False)
    