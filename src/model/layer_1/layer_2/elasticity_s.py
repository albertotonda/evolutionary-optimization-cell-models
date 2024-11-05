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
class Sub_Elasticity_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.__df = pd.DataFrame()
        self.thermo = pd.DataFrame()
        self.enzyme = pd.DataFrame()
        self.regulation = pd.DataFrame()

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self.__df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return self.__df.shape

    #################################################################################
    #########        Setter to change the elasticities matrix              ##########
    # For the E_s matrix
    @property
    def df(self):
        if (
            self.thermo.eq(0).all().all()
            and self.enzyme.eq(0).all().all()
            and self.regulation.eq(0).all().all()
        ):
            return self.__df
        else:
            self.__df = self.thermo - self.enzyme + self.regulation
            return self.__df

    @df.setter
    def df(self, matrix):

        # If the new matrix is a np one
        

        if isinstance(matrix, np.ndarray):
            
            # If the new matrix don't have the same shape as the previous one
            if matrix.shape != self.df.shape:
                # Then we report an error
                raise IndexError(
                    "The shape of your input matrix isn't matching with the elasticity matrix"
                )
            # Else, we atribute the value of the np matrix to the elasticity dataframe
            else:

                self.__df.values[:] = matrix
                # And we reset the value of the Jacobian (and everything downstream with a waterfall effect)

                self.__class_MODEL_instance._reset_value(session="E_s")

        # If the new matrix is a dataframe
        elif isinstance(matrix, pd.DataFrame):
            
            # We attribute this dataframe as the new elasticity matrix
            self.__df = matrix

            # And we reset the value of the Jacobian (and everything downstream with a waterfall effect)
            self.__class_MODEL_instance._reset_value(session="E_s")

        # If the new matrix is neither a np or pd one, we report an error
        else:
            raise TypeError(
                "Please enter a numpy matrix or Pandas dataframe to fill the E_s matrix"
            )

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def reset(self):
        ### Description of the fonction
        """
        Method to reset the value of the elasticity E_s and sub_elasticities
        """
        # Reset of the sub_elasticity dataframe
        self.thermo.fillna(0, inplace=True)
        self.enzyme.fillna(0, inplace=True)
        self.regulation.fillna(0, inplace=True)

        # Reset of the main elasticity dataframe
        self.df.fillna(0, inplace=True)
        # Reset of the value of the system
        self.__class_MODEL_instance._reset_value(session="E_s")


    #################################################################################
    #########        Fonction to change a coefficient of the matrix        ##########
    def change(self, flux_name: str, metabolite_name: str, value: float):
        if flux_name not in self.df.index:
            raise NameError(f"The flux name '{flux_name}' is not in the model")
        elif metabolite_name not in self.df.columns:
            raise NameError(
                f"The parameter name '{metabolite_name}' is not in the model"
            )
        else:
            self.df.at[flux_name, metabolite_name] = value
            self.__class_MODEL_instance._reset_value(session="E_s")

    #################################################################################
    #########        Fonction to change the matrix from vectors       ##########
    def change_from_vector(self, vec_values, vec_coordonate=None):

        if vec_coordonate is None :
            vec_coordonate = self.__class_MODEL_instance.MOO.vectors["coordinates"]
        
        vec_values = self.__class_MODEL_instance.MOO.vectors["sign"] * ( (self.__class_MODEL_instance.MOO.vectors["max"]-self.__class_MODEL_instance.MOO.vectors["min"])*vec_values + self.__class_MODEL_instance.MOO.vectors["min"] )
        if isinstance(vec_coordonate[0][0], (int, np.integer)) and isinstance(vec_coordonate[0][1], (int, np.integer)):  
            for (row,column), value in zip(vec_coordonate, vec_values) :
                self.df.iloc[row, column] = value
        else : 
            for (index,column), value in zip(vec_coordonate, vec_values) :
                self.df.loc[index, column] = value

        self.__class_MODEL_instance._reset_value(session="E_s")
    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def half_satured(self, we_return=False):
        ### Description of the fonction
        """
        Method to attribute to the E_s matrix the value of a half-satured enzyme
        """
        self.reset()
        self.__df = -0.5 * self.__class_MODEL_instance.N_without_ext.transpose()

        self.norm_GR()

        if we_return :
            return(self.df)

    #################################################################################
    #########        Fonction nromalize the growth rate fluxes             ##########    
    def norm_GR(self, df=False):
        """
        df : pd.Dataframe
        You can normalise a dataframe that isn't the model
        """
        max_meta_implied = 6
        if df==False :
            for index, row in self.df.iterrows():
                if row[(row != 0)].count() > max_meta_implied :
                    sum_ela = row.abs().sum()
                    self.df.loc[index] = row / sum_ela
        
        elif isinstance(df, pd.DataFrame()) : 
            for index, row in df.iterrows():
                if row[(row != 0)].count() > max_meta_implied :
                    sum_ela = row.abs().sum()
                    df.loc[index] = row / sum_ela

    #################################################################################
    #########        Fonction to update the elasticities matrix            ##########
    def fill_sub_elasticity(self, a=1, b=1):
        ### Description of the fonction
        """
        Method to fill the sub elasticities dataframes
        """
        #self.reset()

        N = self.__class_MODEL_instance.N.to_numpy()

        # Definition of the sub_elasticity of the thermodynamic effects
        M_plus = M_moins = np.zeros(N.shape)

        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                if N[i][j] > 0.0:
                    M_moins[i][j] = np.abs(N[i][j])

                if N[i][j] < 0.0:
                    M_plus[i][j] = np.abs(N[i][j])

        M_moins = np.transpose(M_moins)
        M_plus = np.transpose(M_plus)

        L, N_red = self.__class_MODEL_instance.Link_matrix

        c_int_df = self.__class_MODEL_instance.metabolites.df.loc[
            self.__class_MODEL_instance.metabolites.df.index.isin(N_red.index)
        ]
        c_int = c_int_df["Concentration"].to_numpy()

        k_eq = self.__class_MODEL_instance.reactions.df[
            "Equilibrium constant"
        ].to_numpy()

        zeta = np.exp(
            np.log(k_eq) - np.dot(np.transpose(N_red.to_numpy()), np.log(c_int))
        )

        first_terme = np.linalg.pinv(
            np.diag(np.transpose(zeta - np.ones((k_eq.shape[0], k_eq.shape[0])))[0])
        )
        second_terme = np.dot(np.diag(zeta), M_plus) - M_moins

        ela_thermo = np.dot(first_terme, second_terme)

        for i, index in enumerate(self.thermo.index):
            for j, column in enumerate(self.thermo.columns):
                self.thermo.at[index, column] = ela_thermo[i][j]

        # Definition of the sub_elasticity of the enzymes effects
        beta = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))
        ela_enzyme = np.multiply(beta, np.transpose(N))

        for i, index in enumerate(self.enzyme.index):
            for j, column in enumerate(self.enzyme.columns):
                self.enzyme.at[index, column] = ela_enzyme[i][j]

        # self.enzyme.values[:,:] = ela_enzyme

        # Definition of the sub_elasticity of the regulations effects
        beta = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))
        alpha = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))

        W_acti = W_inib = np.ones(np.shape(np.transpose(N)))

        ela_regu = np.multiply(alpha, W_acti) - np.multiply(beta, W_inib)

        # self.regulation.values[:,:] = ela_regu

        for i, index in enumerate(self.regulation.index):
            for j, column in enumerate(self.regulation.columns):
                self.regulation.at[index, column] = ela_regu[i][j]

        # We reset the value of the Jacobian
        self.__class_MODEL_instance._reset_value(session="E_s")
    #################################################################################
    #########     Fonction to get the value of the elasticity coeff        ##########
    def get_value(self, flux:str, metabolite:str) -> float : 
        ### Description of the fonction
        """
        Fonction to remove an enzyme to the model

        Parameters
        ----------

        flux       : str
            Name of the flux \n
        
        parameter  : str
            Name of the metabolite
        """
        if flux not in self.df.index :
            raise NameError(f"The flux name {flux} is not in the elasticity matrix\n")
        if metabolite not in self.df.columns :
            raise NameError(f"The parameter name {metabolite} is not in the model\n")
        
        return self.df.at[flux, metabolite]