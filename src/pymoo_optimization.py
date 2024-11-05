# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:31:56 2024

Using pymoo to optimize Arthur's model.

@author: Alberto
"""

# instantiate MODEL class: model = MODEL()
# call model.MOO.build_model()
# shape of the vectors to be optimized: model.MOO.vectors["shape"]
# values of the vectors in [0,1]
# apply the vector to the model, model.elasticity.s.change_from_vector(individual)
# get the fitness function, model.MOO.list_fitness()

# imports


# import of a local module
from model.cell_model import MODEL

if __name__ == "__main__" :
    
    # hard-coded values
    population_size = 100
    offspring_size = 100
    max_generations = 1000
    
    # instantiate model
    model = MODEL()
    # initialize model with default internal values for elasticity matrix
    model.MOO.build_model()
    # this is just a printout to check that everything is in order
    print(model.MOO.vectors)