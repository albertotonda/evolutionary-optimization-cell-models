#####################
# Library
#####################
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL


from .layer_2.elasticity_s import Sub_Elasticity_class
from .layer_2.elasticity_p import Elasticity_p_class


#####################
# Class Elasticities
#####################
class Elasticity_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.s = Sub_Elasticity_class(class_MODEL_instance)
        self.p = Elasticity_p_class(class_MODEL_instance)

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self.s.df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return len(self.s.df.shape)
