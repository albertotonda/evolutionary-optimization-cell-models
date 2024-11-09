#####################
# Library
#####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL

#####################
# Class MOO
#####################
class MOO_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.__cache_modified_elements = None
        self.__cache_vectors = None

        # Fake real data of the model
        self.real_data = {"Elasticity" : np.array([]),"Correlation" : pd.DataFrame()}

        self.result = None

    #################################################################################
    ###########           Return the Dataframe of the data               ############
    def __repr__(self) -> str:
        return str(self.__cache_modified_elements)

    #################################################################################
    #########     Fonction to return the number of sampled elements        ##########
    @property
    def len(self):
        return len(self.__cache_modified_elements)


    ##########################################################################################################
    #########      function that add as studied elements the elasticity with respect of N           ##########
    def sampled_elements(self):
        """
        function that add in .modified_elements the elasticity with respect of N
        """

        # If the cache of the sampled elasticity is empty
        if self.__cache_modified_elements == None :
            
            # We attribute to it a dictionnary
            self.__cache_modified_elements = {}

            ela_half = self.__class_MODEL_instance.elasticity.s.half_satured(we_return=True).copy()
            # We look for the half-saturated elasticity coefficent between every internal-species and reactions
            for react in ela_half.index :
                for meta in ela_half.columns :
                    # If this elasticity coefficient isn't 0 :
                    if ela_half.at[react, meta] != 0 :
                        # We add the pair reaction/metabolite as key of the dictionnary and the value
                        self.__cache_modified_elements[(react, meta)] = ela_half.at[react, meta]


    @property
    def modified_elements(self):
        return self.__cache_modified_elements

    ##########################################################################
    #########      function to set the vector for the MOO           ##########
    def set_vector(self) :
        # Initialisation of the dict that contain all the vectors
        vectors = {"shape": 0,
                   "labels": np.array([]),
                   "coordinates": np.array([]),
                    "min": np.array([]),
                    "max": np.array([]),
                    "sign": np.array([]),
                    "mu" : np.array([]),
                    "sigma": np.array([]),
                    "interactions": np.array([])}
        
        vectors["shape"] = len(self.modified_elements)
        
        labels = []
        coordinates = []
        min = []
        max = []
        sign = []
        mu = []
        sigma = []
        for key, value in self.modified_elements.items() :

            labels.append(key)

            id_row = self.__class_MODEL_instance.elasticity.s.df.index.get_loc(key[0])
            id_col = self.__class_MODEL_instance.elasticity.s.df.columns.get_loc(key[1])
            coordinates.append((id_row, id_col))

            min.append(0.)
            #max.append(2*np.abs(value))
            max.append(1)
            sign.append(value/np.abs(value))

            mu.append(np.abs(value))
            sigma.append(0.5)


        vectors["labels"] = np.array(labels)
        vectors["coordinates"] = np.array(coordinates)
        vectors["min"]   = np.array(min)
        vectors["max"]   = np.array(max)
        vectors["sign"]  = np.array(sign)
        vectors["mu"]    = np.array(mu)
        vectors["sigma"] = np.array(sigma)

        self.__cache_vectors = vectors
    
    @property
    def vectors(self):
        return(self.__cache_vectors)


    ###############################################################################################################
    #########      Function to create a false correlation matrix and the elasticity associated           ##########
    def set_real_data(self, rho_matrix = None, random_seed=None) :
        ### Description of the fonction
        """
        Fonction to create fake real data 
        """
        # If the user adds as input a correlation matrix, we will use this one as real data result
        if rho_matrix is not None :
            ela = None

        
        # Otherwise, we create a fake one with random numbers
        else :
            
            # since we are generating pseudo-random number, let's check if the
            # random seed has been specified; in that case, we set up the random
            # number generator from numpy to have repeatable experiments
            if random_seed is not None :
                np.random.seed(random_seed)

            # We take into memory the old value of the elasticity matrix of metabolite
            old_ela = self.__class_MODEL_instance.elasticity.s.df.copy()

            # We create a new one that will serve to compute the new correlation matrix
            ela = self.__class_MODEL_instance.elasticity.s.df.copy()
            ela.values[:] = 0 # We put every coefficents to 0

            # For every key of the modified elements
            for (react, meta), value in self.__cache_modified_elements.items():
                # We look for the max value
                max_value = self.__cache_modified_elements[(react, meta)]
                # Then we create a random number between [0,1] (beta distribution centred on 0.5) and multiply it by the max value (with the right sign)
                rand_value = max_value*np.random.beta(a=5, b=5)

                # We modify the DataFrame 'ela' at the position (react, meta)
                ela.at[react, meta] = rand_value

            # We modify the value of the elasticity matrix E_s of the current model 
            self.__class_MODEL_instance.elasticity.s.df = ela
            self.__class_MODEL_instance.elasticity.s.norm_GR()

            # And compute the correlation matrix with this elasticity and add both matrix into memory
            self.real_data["Elasticity"] = self.__class_MODEL_instance.elasticity.s.df.copy()
            self.real_data["Correlation"] = self.__class_MODEL_instance.correlation.copy()

            # And we reattribuate the previous matrix of the elasticity
            self.__class_MODEL_instance.elasticity.s.df = old_ela



    ##################################################################################################################################
    #########      Function to pick-up random interaction from the correlation matrix and put a mask to only study them     ##########
    def put_random_mask(self, n:int) :
        
        # First we pick n random elements from the dataframe of the true result
        random_indices = np.random.choice(self.real_data["Correlation"].size, size=n, replace=False)
        rows, cols = np.unravel_index(random_indices, self.real_data["Correlation"].shape)

        # We specify in the vectors dictionnary wich elements are studied 
        self.vectors["interactions"] = zip(rows, cols)

        # We create a dataframe full of False
        df_masked = pd.DataFrame(False, index=self.real_data["Correlation"].index, columns=self.real_data["Correlation"].columns)
        # Then we replace the random picked elements by their value
        df_masked.values[rows, cols] = self.real_data["Correlation"].values[rows, cols]

        # And we finally change the correlation matrix with
        self.real_data["Correlation"] = df_masked
    
                
    ################################################################################################################################
    ################################################################################################################################
    ##############  FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS ##############
    ################################################################################################################################
    ################################################################################################################################

    ##################################################################################################################################
    #########      Function that return the fitness between the real data correlation matrix and the current one            ##########
    def fitness(self, a=1) :
        ### Description of the fonction
        """
        Euclidian norm\n
        """

        real_rho = self.real_data["Correlation"]

        current_rho = self.__class_MODEL_instance.correlation.loc[real_rho.index, real_rho.columns]

        # Creation of a mask to select (True) only the elements float in real_rho (cas particulier pour les bool qui sont convertis automatiquement en int)
        mask_float = real_rho.applymap(lambda x: isinstance(x, (float, int)) and not isinstance(x, bool))

        # We look for the difference matrix, but only where the elements are float or int
        diff_rho = (real_rho[mask_float] - current_rho[mask_float]).to_numpy(dtype=np.float64)

        # Remplacement of the NaN by 0
        diff_rho = np.nan_to_num(diff_rho)

        norm = np.linalg.norm(diff_rho, ord=2)
        
        # a*(x-0.5)²
        fitness = np.power(norm-0.5, 2)*a  
        
        return(fitness)

    ##################################################################################################################################
    #########      Function that return the similarity between the real data correlation matrix and the current one         ##########       
    def similarity(self) :
        ### Description of the fonction
        """
        Euclidian norm\n
        """
        
        real_rho = self.real_data["Correlation"]

        current_rho = self.__class_MODEL_instance.correlation.loc[real_rho.index, real_rho.columns]

        # Creation of a mask to select (True) only the elements float in real_rho (cas particulier pour les bool qui sont convertis automatiquement en int)
        mask_float = real_rho.applymap(lambda x: isinstance(x, (float, int)) and not isinstance(x, bool))

        # We look for the difference matrix, but only where the elements are float or int
        diff_rho = (real_rho[mask_float] - current_rho[mask_float]).to_numpy(dtype=np.float64)

        # Remplacement of the NaN by 0
        diff_rho = np.nan_to_num(diff_rho)

        # L1 is more sensible to the global difference
        #norm_L1 = np.abs(diff_rho).sum().sum()  
        # L2 is more usefull to focus on magnitude of difference
        norm_L2 = (diff_rho**2).sum().sum()

        return(norm_L2)
    


    ##################################################################################################################
    #########      Function that return the difference between the current elasticity and the prior         ##########       
    def weighted_norm(self) :
        ### Description of the fonction
        """
        A simple way to give greater weight to larger elements is to multiply the elements of the difference by their absolute value, or by an increasing function of the values, before calculating the norm.
        """
        
        real_rho = self.real_data["Correlation"]

        current_rho = self.__class_MODEL_instance.correlation.loc[real_rho.index, real_rho.columns]

        # Creation of a mask to select (True) only the elements float in real_rho (cas particulier pour les bool qui sont convertis automatiquement en int)
        mask_float = real_rho.applymap(lambda x: isinstance(x, (float, int)) and not isinstance(x, bool))

        # We look for the difference matrix, but only where the elements are float or int
        diff_rho = (real_rho[mask_float] - current_rho[mask_float]).to_numpy(dtype=np.float64)

        # Remplacement of the NaN by 0
        diff_rho = np.nan_to_num(diff_rho)

        # Calculation of the pondered differences
        diff_rho_weighted = (diff_rho**2) * np.abs(diff_rho)

        # calculation of the pondered norm
        norm_weighted = np.sqrt(diff_rho_weighted.sum().sum())

        return(norm_weighted)
    

    ##################################################################################################################################
    #########      Function evaluate the difference between the current elasticity and the one that should be use           ##########       
    def prior_divergence(self) :
        ### Description of the fonction
        """
        Function evaluate the difference between the current elasticity and the one that should be use
        """
        sum = 0
        for i, (react, meta) in enumerate(self.vectors["labels"]) :
            ela = self.__class_MODEL_instance.elasticity.s.df.at[react, meta]

            sum+= ((np.abs(ela) - np.abs(self.vectors["mu"][i]) )**2 )/(self.vectors["sigma"][i]**2)
            

        return(sum)
    
    ##################################################################################################################################
    #########      Function evaluate the difference between the current elasticity and the one that should be use           ##########       
    def prior_divergence2(self) :
        ### Description of the fonction
        """
        Log diff between the mean and the current value of the elasticity
        """
        sum = 0
        for i,(react, meta) in enumerate(self.vectors["labels"]) :
            # Mean value
            mu = self.vectors["mu"][i]
            # Current value
            x = self.__class_MODEL_instance.elasticity.s.df.at[react, meta]

            sum += -np.log(1-np.abs(x-mu)/mu)

        return(sum)

    ##################################################################################################
    #########      Function to return the fitness value that we are interested in           ##########      
    def list_fitness(self):
        ### Description of the function
        """
        Function to return the fitness value that we are interested in
        """
        return([self.similarity(), self.prior_divergence()])


    ##########################################################################
    #########         Function to build a premade model             ########## 
    def build_model(self, source_file=None, random_seed=None):
        """
        Big : bool
            Do we create a big model of E.Coli Core ? 
            else a small linear model is created

        random_seed : int
            Seed for generated the fake real data
        """
        if source_file is not None :
            self.__class_MODEL_instance.read_SBtab(filepath=source_file)
        else :
            self.__class_MODEL_instance.creat_linear(4, grec=False)
        
        self.__class_MODEL_instance.enzymes.add_to_all_reaction()
        self.__class_MODEL_instance.parameters.add_externals()
        self.__class_MODEL_instance.parameters.add_enzymes()
        self.__class_MODEL_instance.parameters.remove("Temperature")
        self.__class_MODEL_instance.elasticity.s.half_satured() 

        self.__class_MODEL_instance.MOO.build_data(random_seed=random_seed)


    ##########################################################################
    #########         Function to build a premade model             ########## 
    def build_data(self, random_seed=None):
        """
        """

        self.sampled_elements()
        self.set_real_data(random_seed=random_seed)
        self.set_vector()


    #######################################################################
    #########      Function to launch the MOO alogorithm         ##########  
    def launch(self, max_evaluations=300, pop_size=100, num_selected=50, print_result=False) :
        """
        max_evaluations : int
        Total number of individual evaluated\n

        pop_size : int
        Number of individual at each generation\n

        num_selected : int
        Number of best individual that we keep at each generation\n

        print_result : bool
        Do we print the result ?
        """
        from Genetic_Algo import main

        main(self.__class_MODEL_instance, print_result, max_evaluations=max_evaluations, pop_size=pop_size, num_selected=num_selected)
    
