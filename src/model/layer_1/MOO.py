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
        self.__cov_is_studied = True
        self.First_article = True

        # Fake real data of the model
        self.real_data = {"Elasticity" : np.array([]),"Correlation" : pd.DataFrame(), "Covariance" : pd.DataFrame()}

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

            ela_half = self.__class_MODEL_instance.elasticity.s.half_satured(returned=True).copy()
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
        
        # For the 1st article
        if self.First_article:

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
                
                sign.append(value / np.abs(value))

                mu.append(np.abs(value))
                sigma.append(0.5)


            vectors["labels"] = np.array(labels)
            vectors["coordinates"] = np.array(coordinates)
            vectors["min"]   = np.array(min)
            vectors["max"]   = np.array(max)
            vectors["sign"]  = np.array(sign)
            vectors["mu"]    = np.array(mu)
            vectors["sigma"] = np.array(sigma)
        
        # For the case of the 2nd article
        else :
            # Initialisation of the dict that contain all the vectors
            vectors = {"shape": 0,
                    "arrow_labels": np.array([]),
                    "min": np.array([]),
                    "max": np.array([]),
                    "sign": np.array([]),
                    "target": np.array([]),
                    "weight": np.array([])}
            
            regulateds = self.__class_MODEL_instance.reactions.list
            regulators = self.__class_MODEL_instance.metabolites.list
            shape = 0
            arrows_labels = []
            min = []
            max = []
            sign = []
            for regulted in regulateds:
                for regulator in regulators:
                    shape+=1
                    arrows_labels.append([regulted, regulator])
                    min.append(0)
                    max.append(1)
                    sign.append((-1)**shape)

            vectors["shape"]   = shape
            vectors["arrow_labels"] = np.array(arrows_labels)
            vectors["min"]   = np.array(min)
            vectors["max"]   = np.array(max)
            vectors["sign"]  = np.array(sign)

        self.__cache_vectors = vectors
    
    @property
    def vectors(self):
        return(self.__cache_vectors)


    ###############################################################################################################
    #########      Function to create a false correlation matrix and the elasticity associated           ##########
    def set_real_data(self, seed, rho_matrix = None) :
        ### Description of the fonction
        """
        Fonction to create fake real data 
        """
        # Seed of the random number generation
        np.random.seed(seed)

        # If the user adds as input a correlation matrix, we will use this one as real data result
        if rho_matrix is not None :
            ela = None

        
        # Otherwise, we create a fake one with random numbers
        else :

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
                rand_value =2*max_value*np.random.beta(a=5, b=5)


                # We modify the DataFrame 'ela' at the position (react, meta)
                ela.at[react, meta] = rand_value

            # We modify the value of the elasticity matrix E_s of the current model 
            self.__class_MODEL_instance.elasticity.s.df = ela
            self.__class_MODEL_instance.elasticity.s.norm_GR()

            # And compute the correlation matrix with this elasticity and add both matrix into memory
            self.real_data["Elasticity"] = self.__class_MODEL_instance.elasticity.s.df.copy()
            self.real_data["Correlation"] = self.__class_MODEL_instance.correlation.copy()
            self.real_data["Covariance"] = self.__class_MODEL_instance.covariance.copy()

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
    
    @property
    def cov_is_studied(self):
        return(self.__cov_is_studied)
    
    @cov_is_studied.setter
    def cov_is_studied(self, boolean):
        self.__cov_is_studied = boolean
    

    ################################################################################################################################
    ################################################################################################################################
    ##############  FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS - FITNESS ##############
    ################################################################################################################################
    ################################################################################################################################


    ##################################################################################################################################
    #########      Function that return the similarity between the real data correlation matrix and the current one         ##########       
    def similarity(self) :
        ### Description of the fonction
        """
        Euclidian norm\n
        """
        if self.__cov_is_studied == True :
            matrix_real_data = self.real_data["Covariance"]
            matrix_model = self.__class_MODEL_instance.covariance.loc[matrix_real_data.index, matrix_real_data.columns]
        else :
            matrix_real_data = self.real_data["Correlation"]
            matrix_model = self.__class_MODEL_instance.correlation.loc[matrix_real_data.index, matrix_real_data.columns]

        

        # Creation of a mask to select (True) only the elements float in real_rho (cas particulier pour les bool qui sont convertis automatiquement en int)
        mask_float = matrix_real_data.applymap(lambda x: isinstance(x, (float, int)) and not isinstance(x, bool))

        # We look for the difference matrix, but only where the elements are float or int
        matrix_diff = (matrix_real_data[mask_float] - matrix_model[mask_float]).to_numpy(dtype=np.float64)

        # Remplacement of the NaN by 0
        matrix_diff = np.nan_to_num(matrix_diff)

        # L1 is more sensible to the global difference
        #norm_L1 = np.abs(matrix_diff).sum().sum()
        
        # L2 is more usefull to focus on magnitude of difference
        #norm_L2 = np.sqrt((matrix_diff**2).sum().sum())
        #np.linalg.norm(matrix_diff, ord='fro')
        return( (matrix_diff**2).sum().sum() )
    

    def similarity2(self, a=1) :
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


    #############################################################################################################
    #########                 Function evaluate the cost of the introduced regulation arrows           ##########  
    def cost_enzyme(self) :
        ### Description of the fonction
        """
        Function to evaluate the cost of the introduced regulation arrows
        """
        # First we get every arrow labels implied in this experience
        label_regulations = [f"{item[0]}&{item[1]}" for item in self.vectors["arrow_labels"]]
        
        # Then we get a list of 0 and 1 that represent if the arrow is activated or not
        int_list = np.array(self.__class_MODEL_instance.regulations.df.loc[label_regulations, 'Activated'].tolist(), dtype=int)

        # Same with the absolute value of the coeffcient
        coeff_list = np.abs(self.__class_MODEL_instance.regulations.df.loc[label_regulations, 'Coefficient of regulation'].tolist(), dtype="float64")
        
        # Then we do the sum of every enzyme cost of each regulation
        sum = np.sum(int_list*coeff_list)

        return(sum)
    
    ##############################################################################################################
    #########                 Function evaluate the benefite of introducing regulation arrows           ##########  
    def cost_output(self) :
        ### Description of the fonction
        """
        Function to evaluate the benefite of introducing regulation arrows 
        """
        list_var = np.array(self.__class_MODEL_instance.variance.loc[self.vectors["target"]]["Variance"].tolist())
        list_weight = np.array(self.vectors["weight"])

        list_prod = list_weight*list_var

        sum = np.sum(list_prod)

        return(sum)

    ##############################################################################################################
    #########                       Function to return the number of arrows                             ##########  
    def number_arrow(self) :
        ### Description of the fonction
        """
        Function to return the number of regulation arrows ON
        """
        number_arrow = self.__class_MODEL_instance.regulations.df["Activated"].sum()

        return(number_arrow)


    ##################################################################################################
    #########      Function to return the fitness value that we are interested in           ##########      
    def list_fitness(self):
        ### Description of the function
        """
        Function to return the fitness value that we are interested in
        """
        if self.First_article:
            return([self.similarity(), self.prior_divergence()])
    
        else :
            return([self.cost_enzyme(), self.cost_output(), self.number_arrow()])
        
    ################################################################################################################################
    ################################################################################################################################
    ##############  DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE - DoE  #################
    ################################################################################################################################
    ################################################################################################################################


    ##########################################################################
    #########         Function to build a premade model             ########## 
    def build_model(self, seed=1 , N=4, first_article=True, source_file=None):
        """
        seed : int
            Seed for the generation of the fake real data\n
        
        N : int > 3 :
            Number of metabolite in the linear model\n
        
        Big : bool
            Do we create a big model of E.Coli Core ? 
            else a small linear model is created\n
        
        Cov_studied : bool
            Do we look for the difference between covariance matrices of the model and the real data ?
            Else, we look for the difference between correlation matrices
        """
        if source_file is not None :
            self.__class_MODEL_instance.read_SBtab(filepath=source_file)
        else :
            self.__class_MODEL_instance.creat_branch(grec=False)
    
        self.__class_MODEL_instance.enzymes.add_to_all_reaction()
        self.__class_MODEL_instance.parameters.add_externals()
        self.__class_MODEL_instance.parameters.add_enzymes()
        self.__class_MODEL_instance.parameters.remove("Temperature")
        self.__class_MODEL_instance.elasticity.s.half_satured() 


        ### Here we specify if the built is for the 1st article or not
        self.__class_MODEL_instance.MOO.First_article = first_article
        
        self.__class_MODEL_instance.MOO.build_data(seed)



    ##########################################################################
    #########         Function to build a premade model             ########## 
    def build_data(self, seed):
        """
        """

        self.sampled_elements()
        self.set_real_data(seed)
        self.set_vector()

        # If it is for the 1st article, we modify the maximum of the elasticity coefficents
        if self.First_article:
            self.vectors["max"] = 2.*np.abs([self.__class_MODEL_instance.elasticity.s.df.loc[row, col] for row, col in self.vectors["labels"]])

    #########################################################################
    ########  Function to interpolate the between the 2 matrices  ###########
    def interpolate(self, N:int):
        if N<2 : 
            raise ValueError(f"N must be an integer egual or higher than 2")
        else :

            # Creation of the real data vector
            vec_real_data = []
            for (flux, meta) in self.vectors["labels"] :
                vec_real_data.append(self.real_data["Elasticity"].at[flux, meta])
            vec_real_data = np.abs(vec_real_data)


            interpolated_vectors = np.array([
                                    np.linspace(start, end, N) for start, end in zip(vec_real_data, self.vectors["sigma"])
                                ])
            
            interpolated_vectors = interpolated_vectors.T

            return(interpolated_vectors)


    #######################################################################
    #########      Function to launch the MOO alogorithm         ##########  
    def launch(self) :
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
        from inspyred_biological_optimization import main

        main(self.__class_MODEL_instance)
    
