# small set of utility functions

import numpy as np

def compute_distance_from_model(model, individuals) :
    """
    Another utility function, to compute distances straight from the model.
    Now, the main obstacle here is that each individual is a vector of reals
    in [0,1], while the elasticity matrix has other values. An individual
    can be converted into the elasticity matrix representation, however.
    """
    distances = np.zeros((len(individuals),))
    
    # copy the original elasticty matrix
    reference_elasticity_matrix = model.MOO.real_data["Elasticity"].values.copy()
    print(reference_elasticity_matrix)
    
    # for each individual, update the model, get the new matrix, compute distance
    for index, individual in enumerate(individuals) :
        model.elasticity.s.change_from_vector(individual)
        individual_elasticity_matrix = model.elasticity.s.df.values
        print(individual_elasticity_matrix)
        
        distances[index] = np.linalg.norm(individual_elasticity_matrix - reference_elasticity_matrix)
    
    return distances

if __name__ == "__main__" :
    """
    This main here is just to test the functions above.
    """
    
    from model.cell_model import MODEL
    
    random_seed = 42
    
    model = MODEL()
    model.MOO.build_model(random_seed=random_seed)
    
    # this is a bit hard-coded, but it should be the exact individual
    # corresponding to the randomly initialized elasticity matrix
    exact_individual = [0.28553827, 0.24999901, 0.28947616, 0.19268772]
    
    individuals = [[0.5] * 4, [0.1, 0.2, 0.3, 0.4], exact_individual]
    
    distances = compute_distance_from_model(model, individuals)
    
    for index, individual in enumerate(individuals) :
        print("Individual: %s -> distance: %.4f" %
              (str(individual), distances[index]))
    
    