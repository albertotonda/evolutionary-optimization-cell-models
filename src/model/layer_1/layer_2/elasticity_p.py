#####################
# Library
#####################
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL
#####################
# Class Sub_Elasticities
#####################
class Elasticity_p_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.__df = pd.DataFrame(columns=["Temperature"])

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self.__df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return self.__df.shape

    # For the E_p matrix
    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, matrix):
        # If the new matrix is a numpy matrix, its shape must be exactly the same as the previous dataframe
        if isinstance(matrix, np.ndarray):
            # If it is not the case, we report an error
            if matrix.shape != self.df.shape:
                raise IndexError(
                    "The shape of your matrix isn't matching with the elasticity matrix"
                )
            else:
                # Else, we atribute the value of the np matrix to the elasticity dataframe
                self.__df.values[:] = matrix
                # And we reset the value of the MCA matrices
                self.__class_MODEL_instance._reset_value(session="e_p")

        # If the new matrix is a dataframe, it's shape must be N_reaction X N_parameters
        elif isinstance(matrix, pd.DataFrame):
            self.__df = matrix
            self.__class_MODEL_instance._reset_value(session="e_p")

        else:
            raise TypeError(
                "Please enter a numpy matrix  or Pandas dataframe to fill the E_s matrix" )

    #################################################################################
    #########        Fonction to change a coefficient of the matrix        ##########
    def change(self, flux_name: str, parameter_name: str, value: float) :
        if flux_name not in self.df.index:
            raise NameError(f"The flux name '{flux_name}' is not in the model")
        elif parameter_name not in self.df.columns:
            raise NameError(
                f"The parameter name '{parameter_name}' is not in the model"
            )
        else:
            self.df.at[flux_name, parameter_name] = value
            self.__class_MODEL_instance._reset_value(session="E_p")

    #################################################################################
    #########          Fonction to add columns to the E_p matrix           ##########
    def add_columns(self, parameters_to_add: list):
        missing_para = [
            para for para in parameters_to_add if para not in self.df.columns
        ]

        if missing_para != []:
            # Creation of a temporary dataframe with the missing column and full of 0
            new_columns = pd.DataFrame(0, index=self.df.index, columns=missing_para)
            # Then we concatenate

            self.df = pd.concat([self.df, new_columns], axis=1)

    #################################################################################
    #########        Fonction to remove columns to the E_p matrix          ##########
    def remove_columns(self, parameters_to_remove: list):
        para_to_remove_from_E_p = [
            para
            for para in parameters_to_remove
            if para not in self.__class_MODEL_instance.parameters.df.index
        ]

        if para_to_remove_from_E_p != []:
            self.df.drop(columns=para_to_remove_from_E_p, inplace=True)


    #################################################################################
    #########     Fonction to get the value of the elasticity coeff        ##########
    def get_value(self, flux:str, parameter:str) -> float : 
        ### Description of the fonction
        """
        Fonction to remove an enzyme to the model

        Parameters
        ----------

        flux       : str
            Name of the flux \n
        
        parameter  : str
            Name of the parameter
        """
        if flux not in self.df.index :
            raise NameError(f"The flux name {flux} is not in the elasticity matrix\n")
        if parameter not in self.df.columns :
            raise NameError(f"The parameter name {parameter} is not in the model\n")
        
        return self.df.at[flux, parameter]