#####################
# Library
#####################
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL

#####################
# Class Metabolites
#####################
class Metabolite_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        # Private list to deal with the fact that a dataframe cannot be filled if there is no collumn in the dataframe
        self.__cache_meta = []

        self.df = pd.DataFrame(columns=["External", "Concentration", "Unit"])

    #################################################################################
    #########           Return the Dataframe of the metabolites            ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of metabolites          ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    #########           Fonction to add a metabolite                         ##########
    def add(self, name: str, external=False, concentration=1.0, unit = "mmol/gDW"):
        ### Description of the fonction
        """
        Fonction to add a metabolite to the model\n
            If it is already in, it change the properties
        
        Parameters
        ----------
        name          : str
            Name of the metabolite\n

        external      : bool
            is the metabolite external ?\n
        
        concentration : float
            Concentration of the metabolite at the reference state\n
        
        unit          : str
            Unit of the concentration\n


        """
        # Look if the metabolite class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(columns=["External", "Concentration", "Unit"])
        
        elif not isinstance(external, bool) :
            raise TypeError(f"The input argument 'external' must be a bool, not a {type(external)}")
    
        elif not isinstance(concentration, (int,float)) :
            raise TypeError(f"The input argument 'concentration' must be a number, not a {type(concentration)}")
        
        elif not isinstance(unit, str) :
            raise TypeError(f"The input argument 'unit' must be a string, not a {type(unit)}")


        # Else, the metabolite is add to the model by an add to the DataFrame
        else:
            if name in self.df.index :
                self.change(name, external, concentration, unit)
            else : 
                self.df.loc[name] = [external, concentration, unit]

                # If there is no reaction in the columns of the soichio metric matrix, we keep in memeory the metabolite
                if self.__class_MODEL_instance.Stoichio_matrix_pd.columns.size == 0:
                    self.__cache_meta.append(name)
                    print("Don't worry, the metabolite will be add after the add of the 1st reaction")

                # Else, we add every metabolite that we keeped into memory to the stoichiometrix matrix
                else:
                    self.__cache_meta.append(name)
                    for meta in self.__cache_meta:
                        if meta not in self.__class_MODEL_instance.Stoichio_matrix_pd.index:
                            self.__class_MODEL_instance.Stoichio_matrix_pd.loc[meta] = 0.0

                    self.__cache_meta = []

    #################################################################################
    #########           Fonction to change a metabolite                    ##########
    def change(self, name: str, external=None, concentration=None, unit=None):
        ### Description of the fonction
        """
        Fonction to change a metabolite properties in the model

        Parameters
        ----------
        name           : str
            Name of the metabolite to change\n

        external       : bool
            Is the metabolite external ?\n
        
        concentration  : float
            Concentration of the metabolites at the reference state\n

        /!\ If you use external metabolite as parameters
        """
        if name not in self.df.index:
            raise NameError(f"The name '{name}' is not in the metabolite dataframe")

        else:
            if external != None:
                if type(external) != bool:
                    raise TypeError(
                        f"The input variable '{external}' is a type '{type(external)}', a boolean True/False is expected for the argument 'External' !"
                    )
                else:
                    # If it was internal and become external => we remove it from the elasticity matrix
                    self.df.at[name, "External"] = external
                    #self.__class_MODEL_instance._update_elasticity()

            if concentration != None:
                if isinstance(concentration, (int, float, complex)):
                    if concentration < 0:
                        raise ValueError(
                            f"The value of the input '{concentration}' must be greater than 0 !"
                        )
                    else:
                        self.df.at[name, "Concentration"] = concentration
                else:
                    raise TypeError(
                        f"The input variable '{concentration}' is a type '{type(concentration)}', a value is expected for the argument 'concentration' !"
                    )
            
            if isinstance(unit, str) :
                self.df.at[name, "Unit"] = unit

    #################################################################################
    #########           Fonction to remove a metabolite                    ##########

    def remove(self, name: str):
        ### Description of the fonction
        """
        Fonction to remove a metabolite to the model
        
        Parameters
        ----------
        name        : str
            Name of the metabolite to remove
        """

        # Look if the metabolite is in the model
        if name not in self.df.index:
            raise NameError(
                f"Please enter a valide name, {name} isn't in the model ! \n")

        else :
            # Else, the metabolite is remove from the dataframe
            self.df.drop(name, inplace=True)

            # Also from the stoichiometric matrix
            self.__class_MODEL_instance.N.drop(name, axis=0, inplace=True)

            # And from every mention of it in the reaction dataframe
            for reaction in self.__class_MODEL_instance.reactions.df.index:
                self.__class_MODEL_instance.reactions.df.loc[reaction, "Metabolites"].pop(name, None)

            # Updating the network
            #self.__class_MODEL_instance._update_network

            # Remove this metabolite from the elasticity matrix E_s
            self.__class_MODEL_instance._update_elasticity()

    #################################################################################
    #########           Fonction to update the meta dataframe              ##########
    def _update(self, name=None, external=False, concentration=1, unit = "mmol/gDW"):
        ### Description of the fonction
        """
        Internal function to update the metabolite dataframe after a change of the stoichiometric matrix
        
        Parameters
        ----------
        name          : str
            Name of the metabolite\n

        external      : bool
            Is the metabolite external ?\n
        
        concentration : float
            Concentration of the metabolite at the reference state

        """
        # Look if the metabolite class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(columns=["External", "Concentration", "unit"])

        # Look if the metabolite is already in the model
        if name not in self.df.index:
            self.df.loc[name] = [external, concentration, unit]
