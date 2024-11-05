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
# Class Sampling
#####################
class Sampling_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        # Sampling file of the model
        self.data = pd.DataFrame(columns=["Name", "Type", "Mean", "Standard deviation", "Distribution"])

        self.result = None

    #################################################################################
    ###########           Return the Dataframe of the data               ############
    def __repr__(self) -> str:
        return str(self.data)

    #################################################################################
    #########     Fonction to return the number of sampled elements        ##########
    @property
    def len(self):
        return len(self.data)



    #############################################################################
    ###################   Function add sampling data    #########################
    def add_data(self, name, type_variable: str, mean=True, SD=1, distribution="uniform"):
        ### Description of the fonction
        """
        Fonction add a new row to the data dataframe

        Parameters
        ----------

        name           : str
            Name of the variable to sample, list of 2 string in the case of elasticity \n

        type_variable  : str
            To make reference to the type of the variable to sample \n

        mean           : float or True
            Mean of the variable, if mean = True, mean take the current value of the variable \n

        SD             : float 
            The standard deviation of the random draw of the variable \n

        distribution   : str
            Type of distribution of the random draw of this variable ("uniform", "normal", "lognormal" or "beta")
        """
        reactions   = self.__class_MODEL_instance.elasticity.s.df.index
        metabolites = self.__class_MODEL_instance.elasticity.s.df.columns
        parameters = self.__class_MODEL_instance.elasticity.p.df.columns


        # Case where the elasticity p is sampled
        if type_variable in {"elasticity_p", "elasticity_s", "elasticity_c", "e_s", "e_c", "e_p", }:
            # If the name is not a list in the case of the elasticity, it's bad
            if not isinstance(name, (list, tuple, set)) :
                raise TypeError(
                    "For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity"
                )

            # If the list have more or less than 2 elements, it's not valide
            elif len(name) != 2:
                raise TypeError(
                    "For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity"
                )

            else:
                # We attribute both elements of the list, the first must be the flux name and second the differential name's
                flux, differential = name
                # We check if the flux is in the model
                if flux not in reactions:
                    raise NameError(f'The flux name "{flux}" is not in the elasticity matrix')

                if type_variable == "elasticity_s" :
                    if differential not in metabolites:
                        raise NameError(
                        f'The differential name "{differential}" is not in the elasticity matrices E_s'
                        )
                    if isinstance(mean, (bool,type(None))):
                        mean = self.__class_MODEL_instance.elasticity.s.df.at[flux, differential]
                
                if type_variable == "elasticity_p" :
                    if differential not in parameters:
                        raise NameError(
                        f'The differential name "{differential}" is not in the elasticity matrices E_p'
                        )
                    if isinstance(mean, (bool,type(None))):
                        mean = self.__class_MODEL_instance.elasticity.p.df.at[flux, differential]


        elif type_variable in ("all_e_s", "every_e_s") :
            True

        # Case where a parameter is sampled
        elif type_variable == "parameter":
            if name not in parameters:
                raise NameError(f'The parameter name "{name}" is not in the parameters dataframe')

            if isinstance(mean, (bool,type(None))):
                mean = self.__class_MODEL_instance.parameters.at[name, "Mean values"]

        # Case where the metabolite concentration is sampled
        elif type_variable in ("metabolite", "concentration"):
            if name not in metabolites:
                raise NameError(f'The metabolite name "{name}" is not in the metabolites dataframe')

            if isinstance(mean, (bool,type(None))):
                mean = self.__class_MODEL_instance.metabolites.at[name, "Concentration"]

        # Case where the flux is sampled
        elif type_variable in ("flux", "reaction") :
            if name not in reactions:
                raise NameError(f'The flux name "{name}" is not in the reactions dataframe')

            if isinstance(mean, (bool,type(None))):
                mean = self.__class_MODEL_instance.reactions.at[name, "Flux"]

        # Case where a enzyme concentration/activity is sampled
        elif type_variable == "enzyme":
            if name not in self.enzymes.df.index:
                raise NameError(f'The enzyme name "{name}" is not in the enzymes dataframe')
            if isinstance(mean, (bool,type(None))):
                mean = self.enzymes.at[name, "Concentration / Activity"]

        else:
            raise NameError(
                f'The type "{type_variable}" is not available \n\nThe type of variable allowed are :\n- elasticity_p\n- elasticity_s\n- parameter\n- metabolite or concentration\n- reaction or flux\n- enzyme'
            )

        # Let's check if the name of the distribution are correct
        distribution_allowed = ["uniform", "normal", "lognormal", "beta"]
        if distribution.lower() not in distribution_allowed:
            print(
                f"The name of the distribution '{distribution}' is not handle by the programme !\n\nHere is the distribution allowed :\n- uniform\n- normal\n- lognormal\n- beta\n\n The default uniform is applied"
            )
            distribution = "uniform"
        

        index = 0
        while index in self.data.index:
            index += 1

        self.data.loc[index] = [name, type_variable, mean, SD, distribution]


    #############################################################################
    #################   Function clear sampling data    #########################
    def clear_data(self):
        ### Description of the fonction
        """
        Fonction remove every rows from the data dataframe
        """
        self.data.drop(self.data.index,inplace=True)

    #############################################################################
    ###################   Function sampled the model    #########################
    def launch(self, N: int, type_result="rho", studied=[], seed_constant=1):
        ### Description of the fonction
        """
        Fonction launch a sampling study, it return (mean matrix, SD matrix, value of specified elements at each iteration)

        Parameters
        ----------

        N              : int
            Number of random draw done for each variable of the .data dataframe \n

        type_result         : str
            matrix returned by the code \n

        studied        : list
            list of tuple that specify wich interaction will be returned by that output itered_studied

        seed_constant  : float
            Seed of our radom draw
        """

        # If the number of sample asked if < 1 = bad
        N = int(N)
        if N < 1:
            raise ValueError("The number of sample must be greater or egual to 1 !")

        # Internal function that define the random draw
        def value_rand(type_samp: str, mean: float, SD: float):
            type_samp = type_samp.lower()
            if type_samp == "uniform":
                deviation = (9 * SD) ** 0.25
                #print(f"{mean} ; {SD} ; {deviation}\n")
                return np.random.uniform(mean - deviation, mean + deviation)

            elif type_samp == "normal":
                return np.random.normal(mean, SD)

            elif type_samp == "lognormal":
                return np.random.lognormal(mean, SD)

            elif type_samp == "beta":
                alpha = (((1 - mean) / ((np.sqrt(SD)) * (2 - mean) ** 2)) - 1) / (2 - mean)
                beta = alpha * (1 - mean)
                return np.random.beta(alpha, beta)

        # We call of a dataframe in order to initialise the variable with the good shape and get the name of the indexs and columns
        if type_result == "MI":
            data_frame = self.__class_MODEL_instance.MI.copy()
        else:
            data_frame = self.__class_MODEL_instance.rho.copy()

        shape = data_frame.shape

        # We look if the input studied elements are in the model
        # Even if the rho and Mi matrix is symetric, we separte row and col to index and columns for the case of the response coefficient
        studied_dict = {}

        for study in studied :
            if len(study) == 1 :
                study = study[0]

        for study in studied :
            # Case where we study the interaction of 1 element with otherones
            if isinstance(study,str):
                if study not in data_frame.index :
                    raise NameError(f"The studied element {study} isn't in the model !")
                # Else, everything is allright, so we add a dataframe to the dictionnary of the studied element
                else :
                    studied_dict[study] = pd.DataFrame(columns=data_frame.columns)

                    
            # Case where we study the specific interaction between 2 elements
            if len(study)==2:
                row, col = study
                if row not in data_frame.index :
                    raise NameError(f"The studied element {row} isn't in the model !")
                elif col not in data_frame.columns :
                    raise NameError(f"The studied element {col} isn't in the model !")
                # Else, everything is allright, so we add a dataframe to the dictionnary of the studied element
                studied_dict[study] = pd.DataFrame(columns=['interaction'])


        # Conditional line to deal with the seed of the random values generator
        # If seed_constant is an int, than we use this int as seed to generate the seed of other random value
        if isinstance(seed_constant, int) :
            np.random.seed(seed_constant)

        # We save the original value of the model
        self.__class_MODEL_instance.save_state()

        # Time Counter
        import time

        start = time.time()

        # Internal function the generate a state of the model and return the resulting matrix
        def generate_sampled_matrix(result=type_result) :
            
            for index in self.data.index:

                # We look for the type of the element
                Type_sampling = self.data.at[index, "Type"].lower()

                # And we generate it's temporary value
                rand_value = value_rand(
                            self.data.at[index, "Distribution"],
                            self.data.at[index, "Mean"],
                            self.data.at[index, "Standard deviation"],
                        )
                
                if Type_sampling in {"elasticity_p", "e_p"}:
                    flux, differential = self.data.at[index, "Name"]
                    self.__class_MODEL_instance.elasticity.p.change(flux, differential,rand_value)

                # Special case where we change every coefficent of the elasticity (beta distribution)
                elif Type_sampling in {"every_e_s", "all_e_s"}:
                    self.__class_MODEL_instance.elasticity.s.fill_sub_elasticity()
                
                elif Type_sampling in {"elasticity_s", "elasticity_c", "e_s", "e_c"}:
                    flux, differential = self.data.at[index, "Name"]
                    self.__class_MODEL_instance.elasticity.s.change(flux, differential,rand_value)

                elif Type_sampling in {"parameter", "p"}:
                    self.__class_MODEL_instance.parameters.change(self.data.at[index, "Name"], mean = rand_value)

                elif Type_sampling in {"metabolite", "meta", "m"}:
                    self.__class_MODEL_instance.metabolites.df.at[self.data.at[index, "Name"], "Concentration"] = rand_value

                elif Type_sampling in {"flux", "fluxes", "f", "reaction", "reactions", "r"}:
                    self.__class_MODEL_instance.metabolites.df.at[self.data.at[index, "Name"], "Flux"] = rand_value

                elif Type_sampling in {"enzyme", "enzymes", "e"}:
                    self.__class_MODEL_instance.metabolites.df.at[self.data.at[index, "Name"], "Concentration / Activity"] = rand_value

            if result == "MI":
                matrix_sampled = self.__class_MODEL_instance.MI.copy()
            else:
                matrix_sampled = self.__class_MODEL_instance.rho.copy()
            
            R = self.__class_MODEL_instance.R.copy()
            
            return(matrix_sampled, R)

        # Internal function that compute the mean and SD matrices without saturation of the internal memory of the computer
        def accumulate_stats(n_iterations=N, matrix_shape=shape):
            mean_matrix = np.zeros(matrix_shape)
            M2_matrix = np.zeros(matrix_shape)
            count = 0


            for _ in range(n_iterations):
                data_frame, R = generate_sampled_matrix()


                matrix = data_frame.to_numpy(dtype="float64")
                count += 1
                
                # We add the values of the current iteration to the studied dataframes
                for key in studied_dict.keys() :
                    if isinstance(key,str) :
                        new_data = data_frame.loc[[key]]
                        new_data.rename(index={key: count})
                        studied_dict[key] = pd.concat([studied_dict[key], new_data], ignore_index=True)
                    else :
                        new_data = pd.DataFrame(data=[[data_frame.at[key[0], key[1]]]], index=[count], columns=['interaction'])
                        studied_dict[key] = pd.concat([studied_dict[key], new_data], ignore_index=True)


                # Update of the mean
                delta = matrix - mean_matrix
                mean_matrix += delta / count
                
                # Update of the M2 to compute the variance
                delta2 = matrix - mean_matrix
                M2_matrix += delta * delta2

            # Calcul de la variance et de l'écart-type
            variance_matrix = M2_matrix / (count - 1) if count > 1 else np.zeros(matrix_shape)
            SD_matrix = np.sqrt(variance_matrix)
    
            return mean_matrix, SD_matrix

        import warnings
        # Ignore the RuntimeWarnings (because of the possible infinite value of MI)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        mean, SD  = accumulate_stats()

        
        # Then we reatribuate the intial state of the model
        self.__class_MODEL_instance.upload_state()

        running_time = time.time() - start
        print(f"running time of the code : {running_time} \nSo {running_time/N} per occurences !")
        
        self.result = (pd.DataFrame(mean, index=data_frame.index, columns=data_frame.columns), pd.DataFrame(SD, index=data_frame.index, columns=data_frame.columns), studied_dict)
        return (self.result)
    
    
    
    
    
    ################################################################################
    #                                                                              #
    # boxplot # heatmap # MI # Escher # plot # correlation # Graphic interface
    #                                                                              #
    ################################################################################
    #                                                                              #
    #                              PLOT OF FIGURE                                  #
    #                                                                              #
    ################################################################################

    #############################################################################
    #####################      Function display       ###########################  
    def display(self, matrices=None, type_result="") :
        ### Description of the fonction
        """
        Fonction plot a map of the mean and SD of final result 

        Parameters
        ----------

        matrices              : tuple
            tuple of 2 matrix (mean,SD) as returned by the .launch fucntion. If matrices == None, it will use .result by default (the latest result of the sampling)  \n

        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if matrices == None :
            matrices = self.result

        mean_matrix = matrices[0].to_numpy()
        std_matrix = matrices[1].to_numpy()

        nrows, ncols = mean_matrix.shape

        # Creation of a custom colormap
        if type_result == "MI":
            custom_map = mcolors.LinearSegmentedColormap.from_list("custom", ["white", "blue"])
        else :
            custom_map = mcolors.LinearSegmentedColormap.from_list("custom", ["red", "white", "blue"])

        plt.figure(figsize=(20, 20))

        # Normalisation of the size of the circle
        max_size = min(1.0 / nrows, 1.0 / ncols) * 1000*4 # to modify in order to adjust it depending of the number of element
        min_size = 0  # Minimal size if the SD is at max
        
        std_matrix = np.nan_to_num(std_matrix, nan=0)
        sizes = max_size - (std_matrix - std_matrix.min()) / (std_matrix.max() - std_matrix.min()) * (max_size - min_size)

        # coordinates for scatter
        x, y = np.meshgrid(np.arange(mean_matrix.shape[1]), np.arange(mean_matrix.shape[0]))

        # 
        ax = plt.gca()
        ax.set_facecolor('lightgrey')  # Background color

        if type_result=="MI":
            # Creation of the flatten matrix of the colors
            flattened_matrix = mean_matrix.flatten()

            # Application of a normalisation that doesn't affect the colorbar
            colors = np.where(np.isnan(flattened_matrix), np.nan, flattened_matrix)

            # Creation of an instance of Normalize that respect the originales values
            norm = mcolors.Normalize(vmin=np.nanmin(colors), vmax=np.nanmax(colors))

            # Flattenization of the matrix of sizes to correpond to the dimension of flattened_matrix
            flattened_sizes = sizes.flatten()

            # Main figure
            sc = plt.scatter(x.flatten(), y.flatten(), s=flattened_sizes, c=flattened_matrix, cmap=custom_map, norm=norm, marker='o', edgecolors='none')

            # Selection of onlythe sizes associated to NaN elements
            nan_sizes = flattened_sizes[np.isnan(flattened_matrix)]

            # Colorisation of the points with NaN in black
            plt.scatter(x.flatten()[np.isnan(flattened_matrix)], y.flatten()[np.isnan(flattened_matrix)], s=nan_sizes, c='black', edgecolors='none')

            # Addition of the colorbar that respect the normalisation
            plt.colorbar(sc)



        else :
            plt.scatter(x, y, s=sizes, c=mean_matrix.flatten(), cmap=custom_map, marker='o', edgecolors='none')
            # Ajout de la barre de couleurs pour les carrés
            plt.colorbar()

        # Affichage du plot
        plt.gca().invert_yaxis()  # Inverse l'axe y pour que l'origine soit en haut
        plt.show()



    #############################################################################
    #####################     Function histogramm     ###########################  
    def plot_histogram(self, data=None, bins=10, bounders=(-2,2)):
        ### Description of the fonction
        """
        Fonction plot an histogramm of the occurance of a studied interaction during the sampling

        Parameters
        ----------

        bins              : int
            Number of division of the space of possible value \n

        bounder           : tuple
            tuple of 2 float, 1 for the min and 1 for the max displayed

        """
        if data == None:
            data = self.result[2]
            
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=bins, range=bounders, edgecolor='black', alpha=0.7)
        
        plt.title("Histogramm of the occurance of the value for every interation")
        plt.xlabel("Portion")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    #############################################################################
    #####################     Function boxplot     ###########################  
    def plot_boxplot(self, data, studied="internal"):
        ### Description of the fonction
        """
        Fonction plot an boxplot of a row of the result

        Parameters
        ----------

        data              : Dataframe
            Dataframe returned by the .launch and stocked in .result[2] \n
        """

        plt.figure(figsize=(12, 6))

        if studied == "internal" :
            data = data[self.__class_MODEL_instance.N_without_ext.index]
        elif studied == "fluxes" :
            data = data[self.__class_MODEL_instance.N.columns]


        # Créer un boxplot pour chaque colonne
        plt.boxplot([data[col] for col in data.columns], labels=data.columns, showfliers=False)

        # Ajouter des titres et des étiquettes
        plt.title('')
        if studied == "internal" :
            plt.xlabel('Internal metabolites')
        elif studied == "fluxes" :
            plt.xlabel('Fluxes')
        
        plt.ylabel('Mutual Information')

        # Pivoter les labels de l'axe des x
        plt.xticks(rotation=80)

        # Afficher le graphique
        plt.tight_layout()  # Pour éviter le chevauchement
        plt.show()


    #############################################################################
    #####################     Function boxplot     ###########################  
    def escher(self, matrices=None, studied="glc__D_e_para", type_result="MI", model_json=None, map_json=None):
        ### Description of the fonction
        """
        Fonction plot the escher map of the mean or SD ofter the sampling

        Parameters
        ----------

        matrices              : Dataframe
            Dataframe of the result 
        """
        type_result = type_result.lower()

        if matrices == None :
            matrices = self.result
        matrix = matrices[0]
        #matrix = matrices[1]

        # Setting of the escher map
        import escher
        from escher import Builder

        escher.rc['never_ask_before_quit'] = True

        # If nothing is taken as inpout for the Escher map, we use the default values
        if model_json == None :
            model_json = self.__class_MODEL_instance.default_JSON
        if map_json == None :
            map_json = self.__class_MODEL_instance.default_Escher

        # Definition of the Escher Builder
        builder = Builder(
            height=600,
            map_name=None,
            model_json = model_json,
            map_json= map_json,)

        dict_value_meta = {}
        dict_value_flux = {}

        # Display of the mutual information
        if type_result == "mi" :

            # Change of the scale of the circles displayed
            builder.metabolite_scale = [
            { 'type': 'value', 'value': 0.0, 'color': 'rgba(100, 100, 100, 0.0)', 'size': 20},
            { 'type': 'max'  ,               'color': 'rgba(  0,   0, 100, 1.0)', 'size': 40} ]
            
            # Change of the scale of the arrows displayed
            builder.reaction_scale = [
            { 'type': 'value', 'value': 0.0, 'color': 'rgba(  0, 0,   0, 1.0)', 'size':  0},
            { 'type': 'max'  ,               'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

            # For every metabolite of the model (even the external one)
            for meta in self.__class_MODEL_instance.metabolites.df.index :
                # If the metabolite is internal
                if not self.__class_MODEL_instance.metabolites.df.at[meta, "External"] :
                    dict_value_meta[meta] = matrix.at[studied, meta]
                else :
                    dict_value_meta[meta] = matrix.at[studied, meta+"_para"]
            
            # For every reaction of the model
            for flux in self.__class_MODEL_instance.reactions.df.index :
                # We add its value of the intersection of the matrix between the flux and the element
                dict_value_flux[flux] = matrix.at[studied, flux]
        
        # Implementation of the value to the escher builder
        builder.metabolite_data = dict_value_meta
        builder.reaction_data = dict_value_flux

        # If some metabolites aren't in the model, then they are display in black
        builder.metabolite_no_data_size = 5
        builder.metabolite_no_data_color = 'rgba(100, 100, 100, 1.0)'

        # If some reactions aren't in the model, then they are not display
        builder.reaction_no_data_size = 5
        builder.reaction_no_data_color = 'rgba(100, 100, 100, 1.0)'

        Builder.reaction_scale_preset

        from IPython.display import display
        display(builder)