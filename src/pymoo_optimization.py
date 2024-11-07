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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# there is a super-annoying FutureWarning appearing everytime the evaluation
# function is called, so this block of code here just mutes it
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import of a local module
from model.cell_model import MODEL


# specific problem class which inherits from pymoo's generic problem class
class CellProblem(Problem) :
    
    cell_model_instance = None
    n_processes = 10
    multi_process_evaluation = False
    
    def __init__(self, n_variables, n_objectives, cell_model_instance, xl=0.0, xu=1.0, n_processes=10, multi_process_evaluation=False) :
        # xl and xu are the lower and upper bounds for each variable
        super().__init__(n_var=n_variables, n_obj=n_objectives, xl=0.0, xu=1.0)
        # store the instance of the cell model, to be used later to compute
        # the fitness values
        self.cell_model_instance = cell_model_instance
        # this is potentially used for multiprocessing
        self.multi_process_evaluation = multi_process_evaluation
        self.n_processes = n_processes
        
        
    def _evaluate(self, x, out, *args, **kwargs) :
        # this evaluation function starts from the assumption that 'x' is actually
        # an array containing all individuals; so we can shape the fitness values
        # numpy array accordingly
        fitness_values = np.zeros((x.shape[0], 1, self.n_obj))
        
        if self.multi_process_evaluation :
            # now, it would be great to start a multi-process or multi-thread
            # evaluation; but the bottleneck here is the model instance ; we would
            # need to create independent copies of the model instance, so that each
            # process can run things independently. Maybe we can do it directly
            # inside a specialize multi-processing function
            with multiprocessing.Manager() as manager :
                temporary_results = manager.list([0.0] * x.shape[0])
                lock = multiprocessing.Lock()
                
                processes = []
                for i in range(0, x.shape[0]) :
                    p = multiprocessing.Process(target=multiprocessing_evaluator, args=(x[i], self.cell_model_instance, i, temporary_results, lock))
                    processes.append(p)
                    p.start()
                
                for p in processes:
                    p.join()
            
            # convert the results inside the shared list to numpy array
            for i in range(0, x.shape[0]) :
                fitness_values[i,0,:] = temporary_results[i]
        
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
    
def multiprocessing_evaluator(individual, model, index, fitness_values, lock) :
    """
    Fitness function invoked during the multi-processing step
    """
    # copy the model
    local_model = copy.deepcopy(model)
    # apply modifications and get fitness values
    local_model.elasticity.s.change_from_vector(individual)
    x_fitness_values = local_model.MOO.list_fitness()
    
    with lock :
        #print(individual, model, index, fitness_values, lock)
        fitness_values[index] = np.array(x_fitness_values)
    
    return
    

if __name__ == "__main__" :
    
    # hard-coded values
    population_size = 1000
    offspring_size = population_size
    max_generations = 1000
    random_seed = 42
    results_folder = "../local" # 'local' is not under version control (git)
    
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
    model.MOO.build_model(random_seed=random_seed)
    # this is just a debug printout to check that everything is in order
    print(model.MOO.vectors)
    example_individual = [0.5, 0.5, 0.5, 0.5]
    model.elasticity.s.change_from_vector(example_individual)
    print(model.MOO.list_fitness())
    
    # and now, we begin for real
    print("Setting up the evolutionary algorithm...")
    
    # get the information that we need
    n_variables = model.MOO.vectors["shape"]
    n_objectives = len(model.MOO.list_fitness())
    
    # let's start with instantiating the problem class
    cell_problem = CellProblem(n_variables, n_objectives, model, multi_process_evaluation=True)
    
    # then, let's set up the algorithm
    algorithm = NSGA2(pop_size=population_size)
    
    # start the run
    print("Starting the evolutionary run, population_size=%d, max_generations=%d" %
          (population_size, max_generations))
    result = minimize(  
                cell_problem,
                algorithm,
                ('n_gen', max_generations), 
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
    