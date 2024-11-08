# small set of utility functions

import numpy as np

def compute_distance_from_ground_truth(individuals, ground_truth) :
    """
    Computes Euclidean distance between a set of individuals and a known ground 
    truth; each is represented as an array of values.
    """
    distances = np.zeros((len(individuals),))
    
    for i in range(0, len(individuals)) :
        distances[i] = np.linalg.norm(ground_truth - individuals[i])
        
    return distances

def compute_distance_from_model(model, individuals) :
    """
    Another utility function, to compute distances straight from the model.
    """
    # get labels of each element in the elasticity matrix, plus the elasticity matrix itself
    labels = model.MOO.vectors["labels"]
    elasticity = model.MOO.real_data["Elasticity"]
    
    # in the 'labels', the first element is the index of the elasticity matrix, 
    # while the second element is the column
    ground_truth = [elasticity[l[1]].loc[l[0]] for l in labels]
    
    # EXCEPT THAT THE VALUES OF EACH INDIVIDUAL HAVE TO BE DE-NORMALIZED!
    
    # compute distances
    distances = compute_distance_from_ground_truth(individuals, ground_truth)
    
    return distances