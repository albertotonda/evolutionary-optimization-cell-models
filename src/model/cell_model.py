"""
Created on Tuesday Oct 22 2024

Script for the creation of metabolic network model and study of transmition of information throught it.

@author: tuturax (Arthur Lequertier)
INRAE, MaIAGE
"""

##################################################################################
#                                                                                #
#                                   Library                                      #
#                                                                                #
##################################################################################

# Computation module
import numpy as np
import sympy

from scipy.linalg import expm
from scipy.sparse import csr_matrix, identity, eye
import random
import pandas as pd
import time


# Graphic interface
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, CheckButtons

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from matplotlib.patches import Patch

# Importation of the module of the sub-Class
from .layer_1.reactions import Reaction_class
from .layer_1.metabolites import Metabolite_class
from .layer_1.parameters import Parameter_class
from .layer_1.elasticities import Elasticity_class
from .layer_1.enzymes import Enzymes_class
from .layer_1.regulation import Regulation_class
from .layer_1.operon import Operon_class
from .layer_1.sampling import Sampling_class
from .layer_1.MOO import MOO_class



#####################
# Class MODEL
#####################
class MODEL:
    #############################################################################
    ########   Class method to creat a model from stochi matrix    ##############
    @classmethod
    def from_matrix(cls, matrix):
        class_instance = cls()
        return class_instance

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self):

        ################################
        #  Call of the sub-Class and definition of private attribute
        ################################

        # Call of reaction Class
        self.__reactions = Reaction_class(self)
        # Call of metabolite Class
        self.__metabolites = Metabolite_class(self)
        # Call of elasticity Class
        self.__elasticities = Elasticity_class(self)
        # Call of parameter Class
        self.__parameters = Parameter_class(self)
        # Call of enzyme Class
        self.__enzymes = Enzymes_class(self)
        # Call of regulation Class
        self.__regulations = Regulation_class(self)
        # Call of operon Class
        self.__operons = Operon_class(self)
        # Call of the sampling Class
        self.__sampling = Sampling_class(self)
        # Call of the MOO Class
        self.__MOO = MOO_class(self)
        


        ################################
        #  Initialisation of few variables of the model
        ################################

        # Initialisation of the stoichiometric matrix attribute
        self.__Stoichio_matrix_pd = pd.DataFrame()
        
        # bool to specify if the update is ON after a modification of the model
        self.__activate_update = True

        # Default value for the escher map and the JSON model
        self.default_Escher = None
        self.default_JSON = None


        # Frequency of the system
        self.__frequency_omega = 0.0

        # Time of computation
        self.__computation_time = {}

        # Growth rate coeff
        self.lambda_gr = 0.0

        # Ration for the study of joint element
        self.__ratio = pd.DataFrame(columns=["Numerator", "Denominator"])

        # Cache of the Network
        self.__cache_Link_matrix = None
        self.__cache_Reduced_Stoichio_matrix = None
        # Cache of the dynamic proprites
        self.__cache_Jacobian = None
        self.__cache_Reversed_Jacobian = None

        # Cache of MCA coefficients
        self.__cache_R_s_p = None
        self.__cache_R_v_p = None
        self.__cache_R_s_c = None
        self.__cache_R_v_p = None
        self.__cache_R = None

        # Cache cache :)
        self.__cache_cov = None
        self.__cache_h = None
        self.__cache_joint_h = None
        self.__cache_rho = None
        self.__cache_MI = None
        self.__cache_conditional_h = None
        
        
        ################################
        #  Message display when the model is created
        ################################
        print(" ")

    #################################################################################
    ######    Representation = the Dataframe of the Stoichiometric matrix     #######
    def __repr__(self) -> str:
        return str(self.Stoichio_matrix_pd)

    #############################################################################
    ##################              Getter                  #####################
    @property
    def Stoichio_matrix_pd(self) -> pd.DataFrame :
        return self.__Stoichio_matrix_pd

    @property
    def Stoichio_matrix_np(self) -> np.ndarray :
        return self.Stoichio_matrix_pd.to_numpy(dtype="float64")

    @property
    def N(self) -> pd.DataFrame :
        return self.Stoichio_matrix_pd

    @property
    def __N(self) -> np.ndarray :
        return self.Stoichio_matrix_np

    @property
    def N_without_ext(self) -> pd.DataFrame :
        N = self.N.copy()

        # We check if every metabolite is external
        for meta in self.Stoichio_matrix_pd.index:
            if self.metabolites.df.at[meta, "External"] == True:
                # And remove this metabolite from the local stoichio matrix N
                N = N.drop(meta)
        return N

    @property
    def __N_without_ext(self) -> np.ndarray :
        return self.N_without_ext.to_numpy(dtype="float64")
    
    @property
    def activate_update(self):
        return self.__activate_update
    
    @activate_update.setter
    def activate_update(self, value):
        if not isinstance(value, bool) :
            raise TypeError("The attribute of the model 'activate_update' must be a bool\n")
        else:
            self.__activate_update = value


    @property
    def frequency_omega(self):
        return self.__frequency_omega

    @frequency_omega.setter
    def frequency_omega(self, omega):
        if omega < 0 or omega == False:
            omega = 0.0
        self.__frequency_omega = omega

    @property
    def metabolites(self):
        return self.__metabolites

    @property
    def reactions(self):
        return self.__reactions

    @property
    def parameters(self):
        return self.__parameters

    @property
    def enzymes(self):
        return self.__enzymes

    @property
    def operons(self):
        return self.__operons

    @property
    def regulations(self):
        return self.__regulations

    @property
    def elasticity(self):
        return self.__elasticities
    
    @property
    def sampling(self):
        return self.__sampling
    
    @property
    def MOO(self):
        return self.__MOO



    @property
    def Link_matrix(self):


        # If the value is None, we recompute the link matrix and the reduced soichiometric matrix
        if (self.__cache_Link_matrix is None) or (self.__cache_Reduced_Stoichio_matrix is None):
            # First, we set to 0 the Jacobian matrix that mostly depend of it
            self.__cache_Jacobian = None
            # Definition of a local stoichio matrix that will change gradually in the property
            N = self.N_without_ext
            # .copy() ?

            # Else we take a look to the dependent row of the stoichio matrix
            dependent_rows = []
            independent_rows = []

            _, independent = sympy.Matrix(N.to_numpy(dtype="float64")).T.rref()

            for i, meta in enumerate(N.index):
                if i in independent:
                    independent_rows.append(meta)
                else:
                    dependent_rows.append(meta)

            # Build of the reduced stoichio matrix
            Nr = N.loc[independent_rows]

            # Then we deduce the link matrix L from Nr and N
            L = np.dot(
                N.to_numpy(dtype="float64"),
                np.linalg.pinv(Nr.to_numpy(dtype="float64")),
            )
            # L.dtype = np.float64

            self.__cache_Link_matrix = L
            self.__cache_Reduced_Stoichio_matrix = Nr
        

        return (self.__cache_Link_matrix, self.__cache_Reduced_Stoichio_matrix)



    @property
    def ratio(self):
        return(self.__ratio)






    ######################################################################################
    #                                                                                    #
    # stoichio_matrix # Jacobian # elasticity # response_coeff # covariance # entropy MI #
    #                                                                                    #
    ######################################################################################
    #                                                                                    #
    #                                    COMPUTATION                                     #
    #                                                                                    #
    ######################################################################################

    # The attibute with __ are the one compute with numpy and aims to be call for other compuation
    # The attribute without it are only the representation of the them on dataframe

    # MCA properties

    ###################
    # Jacobian
    @property  # Core
    def __Jacobian(self) :


        if self.__cache_Jacobian is None:

            self.__computation_time = {}
            t_0 = time.time()
            
            # Reset of the cache value of the inversed matrix of J
            self.__cache_Reversed_Jacobian = None
            # Reset of the cache value of the MCA coeff
            self.__cache_R_s_p = None
            self.__cache_R_v_p = None
            self.__cache_R_s_c = None
            self.__cache_R_v_p = None
            self.__cache_R = None
            

            t_0 = time.time()
            # Compute of the J matrix
            L, Nr = self.Link_matrix
            self.__computation_time["Link Matrix computation"] =  time.time() - t_0

            t_0 = time.time()
            # Conversion of the matrix to the scipy sparse matrix format
            L = csr_matrix(L)
            Nr = csr_matrix(Nr.to_numpy(dtype="float64"))
            E_s = csr_matrix(self.elasticity.s.df.to_numpy(dtype="float64"))
            n = Nr.shape[0]
            Id = identity(n, format='csr')


            self.__cache_Jacobian = Nr.dot(E_s).dot(L) - self.lambda_gr * Id

            # Case of a frequency response
            if self.frequency_omega != 0:
                self.__cache_Jacobian = self.__cache_Jacobian - eye(self.__cache_Jacobian.shape[0], dtype=complex) * self.frequency_omega * 1j

            self.__computation_time["Jacobian computation"] =  time.time() - t_0
        
        return self.__cache_Jacobian.toarray()

    @property  # Displayed
    def Jacobian(self):
        Nr = self.Link_matrix[1]
        return pd.DataFrame( self.__Jacobian, index=Nr.index, columns=Nr.index)


    #########################
    # Inverse of the Jacobian
    @property  # Core
    def __Jacobian_reversed(self):

        t_0 = time.time()

        if self.__cache_Reversed_Jacobian is None:

            # Compute the J-1 matrix
            J_inv = np.linalg.pinv(self.__Jacobian)
            #J_inv = np.linalg.pinv(self.__Jacobian - self.lambda_gr )

            # Then we attribute to the cache value of the link matrix the new value
            self.__cache_Reversed_Jacobian = J_inv

        self.__computation_time["Jacobian Reversion"] =  time.time() - t_0

        return self.__cache_Reversed_Jacobian

    @property  # Displayed
    def Jacobian_reversed(self):
        return pd.DataFrame(
            self.__Jacobian_reversed,
            index=self.Jacobian.columns,
            columns=self.Jacobian.index,
        )

    ##############################################
    # Response coefficient : MCA
    
    # Sub-Matrices responses

    # R_s_p
    @property  # Core
    def __R_s_p(self):

        if self.__cache_R_s_p is None:
            
            # reset of the total R matrix
            self.__cache_R = None
            
            E_p = self.elasticity.p.df.copy().to_numpy(dtype="float64")
            
            try:
                # We try to resolve the problem
                L_link = np.linalg.solve(self.__Jacobian, self.Link_matrix[1])
            except np.linalg.LinAlgError:
                # If the matrix is singular 
                L_link = np.linalg.lstsq(self.__Jacobian, self.Link_matrix[1], rcond=None)[0]

            C = -np.dot(
                self.Link_matrix[0], L_link
                        )
            
            R_s_p = np.dot(C, E_p)

            self.__cache_R_s_p = R_s_p


        return self.__cache_R_s_p

    @property  # Displayed
    def R_s_p(self):
        # We only add the internal metabolite as variable
        return pd.DataFrame(
            self.__R_s_p,
            index=self.elasticity.s.df.columns,
            columns=self.elasticity.p.df.columns,
        )

    # R_v_p
    @property  # Core
    def __R_v_p(self):
        
        if self.__cache_R_v_p is None:
            # reset of the total R matrix
            self.__cache_R = None

            self.__cache_R_v_p = np.dot(
                self.elasticity.s.df.to_numpy(dtype="float64"), self.__R_s_p
            ) + self.elasticity.p.df.to_numpy(dtype="float64")
        


        return self.__cache_R_v_p

    @property  # Displayed
    def R_v_p(self):
        return pd.DataFrame(
            self.__R_v_p,
            index=self.elasticity.s.df.index,
            columns=self.elasticity.p.df.columns,
        )

    # R_s_c
    @property  # Core
    def __R_s_c(self):

        if self.__cache_R_s_c is None:
            # reset of the total R matrix
            self.__cache_R = None

            self.__cache_R_s_c = -np.dot(
                np.linalg.solve(self.__Jacobian , self.Stoichio_matrix_np), 
                self.elasticity.s.df.to_numpy(dtype="float64"),
                )
            + np.identity(len(self.Stoichio_matrix_np))

        return self.__cache_R_s_c

    @property  # Displayed
    def R_s_c(self):
        # We only add the internal metabolite
        index_meta = []
        for meta in self.metabolites.df.index:
            if self.metabolites.df.at[meta, "External"] == False:
                index_meta.append(meta)

        return pd.DataFrame(
            self.__R_s_c,
            index=index_meta,
            columns=self.metabolites.df.index)

    # R_v_c
    @property  # Core
    def __R_v_c(self):
        if self.__cache_R_v_c is None:
            # reset of the total R matrix
            self.__cache_R = None

            self.__cache_R_v_c = np.dot(self.elasticity.s.df.to_numpy(dtype="float64"), self.__R_s_c)
        return self.__cache_R_v_c

    @property  # Displayed
    def R_v_c(self):
        return pd.DataFrame(
            self.__R_v_c,
            index=self.reactions.df.index,
            columns=self.metabolites.df.index,
        )

    ##########################
    # Big matrix of response R
    @property  # Core
    def __R(self):

        t_0 = time.time()
        
        R_s_p = self.__R_s_p
        R_v_p = self.__R_v_p

        # R is block matrix of the sub-response matrix
        if self.__cache_R is None:
            
            # We reset the matrices of interaction (cov, rho MI etc...)
            self._reset_value("cov")
            # We reset the value of the correlation
            self.__cache_cov = None

            # And we create the total matrix
            R = np.block([[R_s_p], [R_v_p]])
            self.__cache_R = R
        
        self.__computation_time["R computation"] = time.time() - t_0

        return self.__cache_R

    @property  # Displayed
    def R(self):
        # We only add the internal metabolite
        index_meta = self.R_s_p.index.to_list()
        index_flux = self.R_v_p.index.to_list()

        return pd.DataFrame(
            self.__R,
            index=index_meta + index_flux,
            columns=self.parameters.df.index,
        )
    
    @property
    def R_normalized(self):
        # First we creat the vector of concentrations and fluxes
        df_c = self.metabolites.df[self.metabolites.df['External'] == False][['Concentration']].rename(columns={'Concentration': 'unified'})
        df_v = self.reactions.df[["Flux"]].rename(columns={'Flux': 'unified'})

        serie_unified = pd.concat([df_c, df_v])
        serie_p = self.parameters.df[["Mean values"]]

        matrix_unified = np.linalg.inv(np.diag(serie_unified['unified'].values))
        matrix_p = np.diag(serie_p['Mean values'].values)

        df_unified = pd.DataFrame(matrix_unified, index=serie_unified.index, columns=serie_unified.index)

        df_p = pd.DataFrame(matrix_p, index=serie_p.index, columns=serie_p.index)

        R_normalized = df_unified.dot(self.R.dot(df_p))


        return(R_normalized)

    @property
    def R_augmented(self):

        # Creation of a identity matrix
        Id_matrix = np.eye(self.R.shape[0])

        # Creation of a list that will contain the added rows
        augmented_rows = []
        
        # Then for each ratio that the user added to this dataframe
        for idx, row in self.ratio.iterrows() :
            # We add a row to this almost identity matrix
            new_row = np.zeros(self.R.shape[0]) 

            # If it is in the "Numerator" column, we add a 1 coeff
            for num in [row['Numerator']]:
                if num in self.R.index:
                    new_row[self.R.index.get_loc(num)] = 1
            # If it is in the "Denominator" column, we add a -1 coeff
            for denom in [row['Denominator']]:
                if denom in self.R.index:
                    new_row[self.R.index.get_loc(denom)] = -1
            
            augmented_rows.append(new_row)

        augmented_matrix = pd.DataFrame(np.vstack([Id_matrix] + augmented_rows),index=list(self.R.index) + list(self.ratio.index), columns=self.R.index)

        R_augmented = augmented_matrix.dot(self.R)     

        return(R_augmented)


    #########################################
    # Standard deviation of parameters vector
    @property
    def Standard_deviations(self):
        return self.parameters.df["Standard deviation"]

    ###################
    # Covariance matrix
    @property  # Core
    def __covariance(self):

        t_0 = time.time()
        # If the cache is empty, we recompute the cov matrix and atribute the result to the cache value

        if self.__cache_cov is None:
            # First we reset the value of the correlation # Just in case it wasn't
            self.__cache_rho = None

            # We create a dataframe that represent the Cov matrix of parameters for an easier manipulation for the case of the operons
            df_cov_dp =  pd.DataFrame(np.diag(self.Standard_deviations**2), 
                                        index=self.Standard_deviations.index, 
                                        columns=self.Standard_deviations.index)
            
            # Then we set the value of the covariance matrix of the parameters depending to the operons 
            for operon in self.operons.df.index:
                # If the operon is activated
                if self.operons.df.at[operon, "Activated"] == True :
                    new_value = self.operons.df.at[operon, "Mixed covariance"]
                    for enzyme_1 in self.operons.df.at[operon, "Enzymes linked"] :
                        for enzyme_2 in self.operons.df.at[operon, "Enzymes linked"] :
                            if enzyme_1 != enzyme_2 :
                                df_cov_dp.at[enzyme_1+"_para", enzyme_2+"_para"] = df_cov_dp.at[enzyme_2+"_para", enzyme_1+"_para"] = new_value



            covariance_dp = df_cov_dp.to_numpy()

            # We get the response matrix as a local variable to avoid a call of function everytime.
            if True :
                R = self.R_augmented.to_numpy()
            else :
                R = self.__R

            matrix_RC = np.dot(R, covariance_dp)

            Cov = np.block(
                [
                    [covariance_dp, np.dot(covariance_dp, np.conjugate(R.T))],
                    [matrix_RC, np.dot(matrix_RC, np.conjugate(R.T))],
                ]
            )

            self.__cache_cov = np.asarray(Cov, dtype=np.float64)

        self.__computation_time["Covariance"] =  time.time() - t_0

        # Then we return the cache value
        return self.__cache_cov

    @property  # Displayed
    def covariance(self):
        # R as a local variable to avoid many call
        index = self.R_augmented.columns.to_list() + self.R_augmented.index.to_list()
        # Return the dataframe of the covariance matrix by a call of it
        return pd.DataFrame(
            self.__covariance,
            index= index,
            columns= index
            )
    
    ###################
    # Variance matrix
    @property
    def variance(self):
        return(pd.DataFrame({"Variance": self.covariance.values.diagonal()}, index = self.covariance.index))

    ###########################
    # Correlation
    @property  # Core
    def __correlation(self):
        if self.__cache_rho is None :

            # If there is not frequency aspect
            if self.__frequency_omega == 0.0 :
                
                v = np.diag(self.__covariance).copy()
                for i in range(len(v)):
                    if v[i] != 0:
                        v[i] = 1/np.sqrt(v[i])
                    else :
                        v[i] = 0
                v_matrix = np.diag(v)
                     
                rho = np.dot(v_matrix, np.dot(self.__covariance, v_matrix)  )

            else : 
                for i in range(rho.shape[0]):
                    for j in range(rho.shape[1]):
                        rho_value = np.real(self.__covariance[i][j]) / (
                            (np.real(self.__covariance[i][i]) * np.real(self.__covariance[j][j])) ** 0.5)
                        if rho_value<1.:
                            rho_value=-1.
                        elif rho_value>1.:
                            rho_value=1.

                        rho[i][j] = rho_value
            

            self.__cache_rho = np.clip(rho, -1, 1)

        return self.__cache_rho

    @property  # Displayed
    def correlation(self):
        return pd.DataFrame(
            self.__correlation,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    # Just an other name for the same thing
    @property
    def rho(self):
        return self.correlation

    ###########################
    # Mutual information matrix
    @property  # Core
    def __MI(self):
        if self.__cache_MI is None:
            
            # We need a tolerance because there is a numerical error due to the inverse of the square in the correlation part
            epsilon = 1e-10 
            
            # Clipping to be sure that the correlation stay in the intervale [0, 1)
            adjusted_correlation = np.clip(1 - self.__correlation**2, epsilon, 1.)


            # If the value of the correlation if close to 1 or -1, we set the MI to inf, else it is the real value
            MI = np.where(np.abs(self.__correlation) >= 1 - epsilon, 
                        np.inf, 
                        -0.5 * np.log(adjusted_correlation))
            
            self.__cache_MI = MI

        return self.__cache_MI

    @property  # Displayed
    def MI(self):
        return pd.DataFrame(
            self.__MI,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    ################
    # Entropy matrix
    @property  # Core
    def __entropy(self):
        if self.__cache_h is None:
            vec_h = []
            const = 0.5*np.log(2 * np.pi * np.e)
            for index in self.covariance.index:
                if self.covariance.at[index, index] == 0. :
                    vec_h.append(-np.inf)
                else :
                    vec_h.append(const + 0.5*np.log(self.covariance.at[index, index]))

            self.__cache_h = vec_h

        return self.__cache_h

    @property  # Displayed
    def entropy(self):
        return pd.DataFrame(self.__entropy, index=self.covariance.index, columns=["Entropy"])

    ######################
    # Joint entropy matrix
    @property  # Core
    def __joint_entropy(self):
        if self.__cache_joint_h is None:
            Cov = self.__covariance
            joint_h = np.zeros(Cov.shape)
            for i in range(Cov.shape[0]):
                for j in range(Cov.shape[1]):
                    if Cov[i][i] * Cov[j][j] - Cov[i][j] * Cov[j][i] <= 1e-16:
                        joint_h[i][j] = np.inf

                    else:
                        joint_h[i][j] = np.log(2 * np.pi * np.e) + 0.5 * np.log(
                            Cov[i][i] * Cov[j][j] - Cov[i][j] * Cov[j][i]
                        )

            self.__cache_joint_h = joint_h

        return self.__cache_joint_h

    @property  # Displayed
    def joint_entropy(self):
        return pd.DataFrame(
            self.__joint_entropy,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    ###########################
    # Conditional entropy
    @property  # Core
    def __entropy_conditional(self):
        if self.__cache_conditional_h is None:

            condi_h = np.zeros(shape=self.covariance.shape)
            
            for i in range(condi_h.shape[0]):
                for j in range(condi_h.shape[1]):
                    condi_h[i][j] = self.__entropy[i] - self.__MI[j][i]

            self.__cache_conditional_h = condi_h

        return self.__cache_conditional_h

    @property  # Displayed
    def entropy_conditional(self):
        return pd.DataFrame(
            self.__entropy_conditional,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )


    ################################
    # Temporal control coefficient
    def temporal_C_s_p(self, t=0.0):
        L, N_r = self.Link_matrix

        return np.dot(
            np.dot(
                np.dot(L, expm(self.Jacobian * t) - np.identity(len(self.Jacobian))),
                self.Jacobian_reversed,
            ),
            N_r,
        )

    def temporal_R_s_p(self, t=0.0):
        return np.dot(self.temporal_C_s_p(t), self.elasticity.p.df.to_numpy(dtype="float64"))

    def temporal_C_v_p(self, t=0.0):
        L, N_r = self.Link_matrix

        return np.dot(
            np.dot(
                np.dot(
                    self.elasticity.s.df.to_numpy(dtype="float64"),
                    np.dot(L, expm(self.Jacobian * t) - np.identity(len(self.Jacobian))),
                ),
                self.Jacobian_reversed,
            ),
            N_r,
        ) + np.identity(self.N.shape[1])

    def temporal_R_v_p(self, t=0.0):
        return np.dot(self.temporal_C_v_p(t), self.elasticity.p.df.to_numpy(dtype="float64"))


    ######################################################################################
    #                                                                                    #
    # find # reset # setter # update # time                                              #
    #                                                                                    #
    ######################################################################################
    #                                                                                    #
    #                                    TOOLS                                           #
    #                                                                                    #
    ######################################################################################


    ###########################################################################
    ############  Function to find where the variable name is  ################
    def find(self, name: str):
        """
        Function to find where a specie is in the model

        name : a string of the name that you want to know where is it
        """
        if name in self.metabolites.df.index:
            return "metabolite"
        elif name in self.reactions.df.index:
            return "reaction"
        elif name in self.parameters.df.index:
            return "parameter"
        else:
            raise NameError(f"The input name '{name}' is not in the model !")

    #############################################################################
    #############   Function reset some the values of the model  ################
    def _reset_value(self, session=""):
        session = session.lower()
        ensemble = {"", "e_s"}
        if session in ensemble :
            # Reset the value of the cache data
            self.__cache_Jacobian = None
            self.__cache_Reversed_Jacobian = None

        ensemble |= {"e_p"}
        if session in ensemble:
            # Reset of the cache value of the MCA coeff
            self.__cache_R_s_p = None
            self.__cache_R_v_p = None
            self.__cache_R_s_c = None
            self.__cache_R_v_p = None

        ensemble |= {"var", "cov"}
        if session in ensemble:
            #self.__cache_Link_matrix = None
            #self.__cache_Reduced_Stoichio_matrix = None
            self.__cache_cov = None
            self.__cache_h = None
            self.__cache_joint_h = None
            self.__cache_rho = None
            self.__cache_MI = None
            self.__cache_conditional_h = None

    #############################################################################
    #############   Function to update after a modification of N  ###############

    # Call the update function when the matrix_Stoichio is modified
    @Stoichio_matrix_pd.setter
    def Stoichio_matrix_pd(self, new_df) :

        # If the input matrix is a dataframe
        if isinstance(new_df, pd.DataFrame) :
            
            # If the previous one and the new one have the same shape, we don't update the model
            if new_df.shape == self.__N.shape :
                self.__Stoichio_matrix_pd = new_df
            # Else we update the model

            else :
                # First we remove all the previous reactions
                for reaction in self.reactions.df :
                    self.reactions.remove(reaction)

                self.__Stoichio_matrix_pd = new_df

                self._update_network()

        # If the update matrix is a numpy array
        elif isinstance(new_df, np.ndarray) :
            if new_df.shape == self.Stoichio_matrix_np.shape :
                self.__Stoichio_matrix_pd.values = new_df
            else :
                raise ValueError(f"In the case of the update of the stoichiometric matrix by a numpy matrix, please be sure that the shape are the same !\n")
        
        # Else, the type is wrong
        else :
            raise TypeError(f"Please, to update the whole stoichiometric matrix, use a Pandas dataframe or a Numpy array with the same shape !\n")
        

    def reset(self):
        ### Description of the function
        """
        Function to reset the model
        """
        self.Stoichio_matrix_pd = pd.DataFrame()
        self.parameters.df.reset_index(inplace=True)
        self.enzymes.df.reset_index(inplace=True)
        self.elasticity.s.df.reset_index(inplace=True)
        self.elasticity.p.df.reset_index(inplace=True)

    def _update_network(self) -> None:
        ### Description of the fonction
        """
        Fonction to update the dataframes after atribuated a new values to the stoichio matrix
        """
        if self.activate_update :

            self.metabolites.__init__(self)
            self.reactions.__init__(self)
            # Deal with the metabolites
            for meta in self.Stoichio_matrix_pd.index:
                self.metabolites.add(meta)

            # Deal with the reactions
            # Loop on every reaction of the stoichiometry matrix
            for reaction in self.Stoichio_matrix_pd.columns:
                # Creation of a dictionnary that will contain every metabolite (as keys) and their stoichiometries coeff (as values)
                dict_stochio = {}

                # We also add the stochiometric coefficent to the dataframe of reaction
                for meta in self.Stoichio_matrix_pd.index:
                    if self.Stoichio_matrix_pd.at[meta, reaction] != 0:
                        dict_stochio[meta] = self.Stoichio_matrix_pd.loc[meta, reaction]

                # Then we add the reaction to the reactions Dataframe
                self.reactions.add(name=reaction, metabolites=dict_stochio)

            # Reset the value of the cache data
            self.__cache_Link_matrix = None
            self.__cache_Reduced_Stoichio_matrix = None
            self.__cache_Jacobian = None

            # We update the elasticities matrix based on the new stoichiometric matrix
            self._update_elasticity()


    #################################################################################
    ############     Function to the elaticities matrix of the model     ############
        
    def _update_elasticity(self):
        ### Description of the fonction
        """
        Function to update the elasticities matrices of the model after a direct modification of the stoichiometric matrix
        or reaction and metabolite dataframes
        """

        t_0 = time.time()
        
        if self.activate_update :
            ###
            # First we check the metabolite
            # For every metabolite in the stoichio matrix (without the external one, they are in the parameter section) :
            meta_int = []
            for meta in self.metabolites.df.index :
                if self.metabolites.df.at[meta, "External"] == False :
                    meta_int.append(meta)

            # We look for the metabolite that isn't in the model
            missing_meta = [meta for meta in meta_int if meta not in self.elasticity.s.df.columns]

            if missing_meta:
                # Creation of temporary DataFrame of the missing columns, filled with 0
                df_temp = pd.DataFrame(0., index=self.elasticity.s.df.index, columns=missing_meta, dtype="float64")
                
                # Concatenation of this dataframe of the missing column with the current DataFrame
                self.elasticity.s.thermo = pd.concat([self.elasticity.s.thermo, df_temp], axis=1)
                self.elasticity.s.enzyme = pd.concat([self.elasticity.s.enzyme, df_temp], axis=1)
                self.elasticity.s.regulation = pd.concat([self.elasticity.s.regulation, df_temp], axis=1)

            # For every metabolite of the E_s elasticity matrix :
            for meta in self.elasticity.s.df.columns:
                # If the metabolite isn't in the stoichio matrix => we remove it from the E_s elasticity matrix
                if meta not in self.N_without_ext.index:
                    self.elasticity.s.thermo.drop(columns=meta, inplace=True)
                    self.elasticity.s.enzyme.drop(columns=meta, inplace=True)
                    self.elasticity.s.regulation.drop(columns=meta, inplace=True)

            # Special case when there is no reaction
            # Pandas doesn't allow to add line before at least 1 column is add
            if self.elasticity.s.df.columns.size != 0:
                for reaction in self.reactions.df.index:
                    if reaction not in self.elasticity.s.df.index:
                        self.elasticity.s.thermo.loc[reaction] = [0. for i in self.elasticity.s.df.columns]
                        self.elasticity.s.enzyme.loc[reaction] = [0. for i in self.elasticity.s.df.columns]
                        self.elasticity.s.regulation.loc[reaction] = [0. for i in self.elasticity.s.df.columns]


            

            ##################################
            # Then we deal with the parameters
            missing_para = []

            # For every parameters
            for para in self.parameters.df.index:
                # If it is not in the E_p matrix
                if para not in self.elasticity.p.df.columns:
                    missing_para.append(para)
            # We add it
            self.elasticity.p.add_columns(missing_para)
    
            para_to_remove_from_E_p = []
            # For every parameters in the E_p elasticity matrix
            for para in self.elasticity.p.df.columns:
                # If the parameters isn't in the parameters dataframe, we remove it from E_p
                if para not in self.parameters.df.index:
                    para_to_remove_from_E_p.append(para)
            self.elasticity.p.remove_columns(para_to_remove_from_E_p)

            # Special case when there is no reaction
            # Pandas doesn't allow to add line before at least 1 column is add
            if self.elasticity.p.df.columns.size != 0:
                for reaction in self.reactions.df.index:
                    if reaction not in self.elasticity.p.df.index:
                        self.elasticity.p.df.loc[reaction] = [0 for i in self.elasticity.p.df.columns]


            self._reset_value()

        self.__computation_time["Update elasticity"] =  time.time() - t_0



    def time(self) :
        df = pd.DataFrame.from_dict(self.__computation_time, orient='index', columns=['Time (s)'])
        return(df)


    ########################################################################################
    #                                                                                      #
    # grouped # MI # fixed # rho # joint # entropy # conditional # MI # entropy            #
    #                                                                                      #
    ########################################################################################
    #                                                                                      #
    #                                       STUDY                                          #
    #                                                                                      #
    ########################################################################################


    #################################################################################
    #########    Function that return the entropy of group of variable   ############
    def group_entropy(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the entropy of group of variable (joint entropy)

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy(dtype="float64")

        # If the groups variables is empty, we return the entropy of every single variables and parameters
        if groups == []:
            return self.entropy

        # Else it means that we study a group of variable
        # If groups is a list, we transform it into a dictionnary
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[f"group_{i}"] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        # For every group
        for key in dictionnary.keys():
            group = dictionnary[key]
            # for every variables of a group
            for variable in group:
                # if the variable is not in the model, we raise an error
                if variable not in Cov_df.index:
                    raise NameError(f"The variables {variable} is not in the covariance matrix !")

        # Initialisation of the MI matrix
        entropy = pd.DataFrame(index=dictionnary.keys(), columns=["Entropy"], dtype=float)

        # For each group (= key of dictionnary)
        for key in dictionnary.keys():
            group = dictionnary[key]
            # We recreate a smaller covariance matrice with only the element of the group
            Cov = Cov_df.loc[group, group].to_numpy(dtype="float64")

            entropy.at[key, "Entropy"] = (len(Cov) / 2) * np.log(2 * np.pi * np.e) + 0.5 * np.log(
                np.linalg.det(Cov)
            )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return entropy

    ###############################################################################################
    #######    Function that return the joint entropy matrix for a group of variable   ############
    def group_joint_entropy(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the joint entropy of a group of variable

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy(dtype="float64")

        # If the groups variables is empty, we return the joint entropy matrix of every single variables and parameters
        if groups == []:
            return self.joint_entropy

        # Else it mean that we study a group of variable
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[str(i)] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionnary.keys():
            group = dictionnary[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(f"The variables {variable} is not in the covariance matrix !")

        # Initialisation of the MI matrix
        joint_entropy = pd.DataFrame(index=dictionnary.keys(), columns=dictionnary.keys(), dtype=float)

        for key1 in dictionnary.keys():
            for key2 in dictionnary.keys():
                # extraction of the list of string
                group1 = dictionnary[key1]
                group2 = dictionnary[key2]

                Cov = Cov_df.loc[group1 + group2, group1 + group2].to_numpy(dtype="float64")

                joint_entropy.at[key1, key2] = (len(group1) + len(group2) / 2) ** np.log(
                    2 * np.pi * np.e
                ) + 0.5 * np.log(np.linalg.det(Cov))

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return joint_entropy

    ###############################################################################################
    #######    Function that return the joint entropy matrix for a group of variable   ############
    def group_conditional_entropy(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the conditional entropy of a group of variable

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy(dtype="float64")

        # If the groups variables is empty, we return the joint entropy matrix of every single variables and parameters
        if groups == []:
            return self.entropy_conditional

        # Else it mean that we study a group of variable
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[str(i)] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionnary.keys():
            group = dictionnary[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(f"The variables {variable} is not in the covariance matrix !")

        # Initialisation of the MI matrix
        conditional_entropy = pd.DataFrame(index=dictionnary.keys(), columns=dictionnary.keys(), dtype=float)

        for key1 in dictionnary.keys():
            for key2 in dictionnary.keys():
                # extraction of the list of string
                group1 = dictionnary[key1]
                group2 = dictionnary[key2]

                # Creating the sub_covariance matrix
                Cov1 = Cov_df.loc[group1, group1].to_numpy(dtype="float64")
                Cov2 = Cov_df.loc[group2, group2].to_numpy(dtype="float64")
                # And the big one
                Cov = Cov_df.loc[group1 + group2, group1 + group2].to_numpy(dtype="float64")

                conditional_entropy.at[key1, key2] = (
                    len(Cov) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov))
                    - len(Cov1) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov1))
                )

                conditional_entropy.at[key2, key1] = (
                    len(Cov) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov))
                    - len(Cov2) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov2))
                )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return conditional_entropy

    #################################################################################
    ############    Function that return the Mutual Inforamtion matrix   ############
    def group_MI(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the Mutual information

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance

        # If the groups variables is empty, we return the mutual information of every single variables and parameters
        if groups == []:
            return self.MI

        # Else it mean that we study a group of variable
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[str(i)] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionnary.keys():
            group = dictionnary[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(f"The variables {variable} is not in the covariance matrix !")

        # Initialisation of the MI matrix
        MI = pd.DataFrame(index=dictionnary.keys(), columns=dictionnary.keys(), dtype=float)

        for key1 in dictionnary.keys():
            for key2 in dictionnary.keys():
                # extraction of the list of string
                group1 = dictionnary[key1]
                group2 = dictionnary[key2]

                Cov_1 = Cov_df.loc[group1, group1].to_numpy(dtype="float64")
                Cov_2 = Cov_df.loc[group2, group2].to_numpy(dtype="float64")
                Cov_3 = Cov_df.loc[group1 + group2, group1 + group2].to_numpy(dtype="float64")

                MI.loc[key1, key2] = 0.5 * np.log(
                    np.linalg.det(Cov_1) * np.linalg.det(Cov_2) / np.linalg.det(Cov_3)
                )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return MI

    ############################################################################################################
    ############    Function that compute the change of distribution after the fixation of values   ############
    def group_entropy_fixed_vector(
        self,
        elements_to_fixe=[],
        elements_to_study=[],
        new_mean_fixed=[],
        return_all=False,
        plot=False,
    ):
        ### Description of the fonction
        """
        Fonction to compute the entropy of a group when a vector parameter is fixed
        return the entropy or the new mean and SD

        elements_to_fixe  : str or list
            a string or a list of string representing the variables/parameter to fixed
            if == [] (by default), nothing is fixed \n

        elements_to_study : str or list
            a string or a list of string representing the variables/parameter to study, 
            if == [] (by defalut), all variables/parametersare study\n

        new_mean_vector   : list
            a list contenning a the new mean of the fixed elements
            If == [] (by default), we take the current mean\n

        
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        # Take the covariance matrix as local variable to avoid a lot of function call (and more clarity).
        Cov_df = self.covariance

        # local function to extract the mean from the model by the name of the element fo the model
        def mean_in_the_model(name):
            if name in self.metabolites.df.index:
                return self.metabolites.df.at[name, "Concentration"]
            elif name in self.reactions.df.index:
                return self.reactions.df.at[name, "Flux"]
            elif name in self.parameters.df.index:
                return self.parameters.df.at[name, "Mean values"]
            else:
                raise NameError(
                    f"The input name '{name}' in the 'fixed_vector' argument is not in the metabolite, reactions or parameters dataframe !"
                )

        ##############################
        # Check of the input variables

        # First we check the elements that the user want to fixe
        # If the fixed_elements list is empty (by default), nothing is fixed
        # If it is a str, we convert it into a list
        if isinstance(elements_to_fixe, str) :
            if elements_to_fixe == "" :
                elements_to_fixe = []
            else :
                elements_to_fixe = [elements_to_fixe]

        # Then we check every fixed elements to see if they are in the model
        for element in elements_to_fixe:
            if element not in Cov_df:
                raise NameError(f"The elements '{element}' in the elements_to_fixe input is not in the model !")

        if not isinstance(elements_to_fixe, list):
            raise type("The input argument 'elements_to_fixe' must be a list of string or a single string for the case of 1 element to fixe !\n")


        # Then we check the elements to study
        # If the groups variables is empty (by default), we study every single variables and parameters
        if elements_to_study == []:
            for index in self.covariance.index:
                elements_to_study.append(index)
        
        # If the element to study is a string, it mean there is only 1 element to study
        elif isinstance(elements_to_study, str):
            elements_to_study = [elements_to_study]

        if not isinstance(elements_to_study, list):
            raise type("The input argument 'elements_to_study' must be a list of string or a single string for the case of 1 element to study !\n")

        
        # Finally, we check what the user input for the mean vector

        # If the mean vector is a single value, we convert it to a list of 1 element
        if isinstance(new_mean_fixed, (float, int)) :
            new_mean_fixed = [new_mean_fixed]

        # We check the type of the fixed mean vector
        if not isinstance(new_mean_fixed, list):
            if isinstance(new_mean_fixed, np.ndarray):
                new_mean_fixed = new_mean_fixed.tolist()
            else : 
                raise TypeError(f"The input argument 'new_mean_vector' must be a list or a numpy vector of number !")

        # We fill a list that contnaing the old value of mean of the fixed vector
        old_mean_fixed = []
        for element in elements_to_fixe:
            old_mean_fixed.append(mean_in_the_model(element))

        # In the case of an empty list (by default), we fill it by the old value of mean
        if new_mean_fixed == []:
            new_mean_fixed = old_mean_fixed

        # Case of the size
        if len(elements_to_fixe) != len(new_mean_fixed) and new_mean_fixed != [] :
            raise ValueError(f"The size of 'elements_to_fixe'={len(elements_to_fixe)} and 'new_mean_fixed'={len(new_mean_fixed)} must be the same !\n")
        
        
        new_mean_fixed = np.array(new_mean_fixed)
        old_mean_fixed = np.array(old_mean_fixed)

        # Creation of an entropy dataframe where will be store the new value of entropy
        entropy_df = pd.DataFrame(columns=["Old H", "New H", "Delta H"])
        for studied in elements_to_study :
            entropy_df.at[studied, "Old H"] = self.entropy.at[studied, "Entropy"]


        ##################
        # Computation

        # We create intermediate matrix
        Cov_ss = Cov_df.loc[elements_to_study, elements_to_study].to_numpy(dtype="float64")
        Cov_ff = Cov_df.loc[elements_to_fixe, elements_to_fixe].to_numpy(dtype="float64")
        Cov_sf = Cov_df.loc[elements_to_study, elements_to_fixe].to_numpy(dtype="float64")
        Cov_fs = Cov_sf.T

        
        # The targeted covariance matrix of the studied elements in the case where there is a fixed vector
        Cov_ss_f = Cov_ss - np.dot(Cov_sf, np.dot(np.linalg.inv(Cov_ff), Cov_fs))
        
        vec_h = []
        for i in range(len(Cov_ss_f)):
            vec_h.append(0.5 * np.log(2 * np.pi * np.e * Cov_ss_f[i, i]) + 0.5)

        # New entropy of the studied elements with the fixed vector
        total_entropy = len(Cov_ss) / 2 * np.log(2 * np.pi * np.e) + np.log(np.linalg.det(Cov_ss_f))
        
        for i,index in enumerate(entropy_df.index) :
            entropy_df["New H"] = vec_h
            entropy_df["Delta H"] = entropy_df["New H"] - entropy_df["Old H"] 


        # If the elements are fixed to an other values that the mean, the mean of the study elements change too !
        delta_mean_study = np.dot(
                            np.dot( Cov_sf, np.linalg.inv(Cov_ff)),
                              (new_mean_fixed - old_mean_fixed) )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        #############
        # return part

        # Special line for return of all variable
        # Creation of dataframe 
        if return_all == True:
            # Intitialisation of the SD dataframe
            SD_df = pd.DataFrame(columns=["Old SD", "New SD", "Delta SD"])
            # Intitialisation of the SD dataframe
            mean_df = pd.DataFrame(columns=["Old mean", "New mean", "Delta mean"])
            # Then we add a line of 0 for each variable study
            for element in elements_to_study:
                SD_df.loc[element] = [0 for column in SD_df.columns]
                mean_df.loc[element] = [0 for column in mean_df.columns]


        # Case where we must return all variable and plot
        if return_all == True or plot == True:
            # We transform the final covariance matrix to dataframe
            Cov_ss_f_df = pd.DataFrame(Cov_ss_f, index=elements_to_study, columns=elements_to_study)

            # And the new mean too
            delta_mean_study_df = pd.DataFrame(delta_mean_study, index=elements_to_study, columns=["Delta"])

            # For every variables/parameters study
            for element in elements_to_study:
                # We add the old value of SD and mean
                SD_df.at[element, "Old SD"] = np.sqrt(Cov_df.at[element, element])
                old_mean_study = mean_in_the_model(element)
                mean_df.at[element, "Old mean"] = old_mean_study

                # Then the new ones after to fixe the vector
                SD_df.at[element, "New SD"] = np.sqrt(Cov_ss_f_df.at[element, element])
                mean_df.at[element, "New mean"] = delta_mean_study_df.at[element, "Delta"] + old_mean_study

                # And we also look for the difference
                SD_df.at[element, "Delta SD"] = np.abs(
                    np.sqrt(Cov_df.at[element, element]) - np.sqrt(Cov_ss_f_df.at[element, element])
                )
                mean_df.at[element, "Delta mean"] = delta_mean_study_df.at[element, "Delta"]

            if plot == True:
                self.boxplot(elements_to_fixe, elements_to_study, new_mean_fixed)

            if return_all == True:
                return SD_df, mean_df

        ##elif return_Cov_and_mean == True:
        ##    return entropy, Cov_ss_f, delta_mean_study

        else:
            return entropy_df

    """
   #################################################################################
    ############    Function that return the Mutual Inforamtion matrix   ############
    def group_entropy_fixed_vector(
        self,
        groups_to_study=[],
        fixed_elements=[],
        new_mean_vector=[],
        return_Cov_and_mean=False,
    ):

        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        # Just the initialisation of locals variables to taking into acount the well computation of the default case
        groups_is_all = False
        groups_is_all = False

        # Take the covariance matrix as local variable to avoid a lot of function call.
        Cov_df = self.covariance

        # local function to extract the mean from the model by the name of the element fo the model
        def mean_in_the_model(name):
            if name in self.metabolites.df.index:
                return self.metabolites.df.at[name, "Concentration"]
            elif name in self.reactions.df.index:
                return self.reactions.df.at[name, "Flux"]
            elif name in self.parameters.df.index:
                return self.parameters.df.at[name, "Mean values"]
            else:
                raise NameError(
                    f"The input name '{name}' in the 'fixed_vector' argument is not in the metabolite, reactions or parameters dataframe !"
                )

        # If the groups variables is empty (by default), we study every single variables and parameters
        if groups_to_study == []:
            groups_is_all = True
            groups_to_study = [[]]
            for index in self.covariance.index:
                groups_to_study[0].append(index)

        # If we just take as input a list of str, we just transform it into a list of list
        elif type(groups_to_study) == list and type(groups_to_study[0]) == str:
            groups_to_study = [groups_to_study]

        # If the fixed_elements list is empty (by default), we fix every single variables and parameters
        if fixed_elements == []:
            fixed_is_all = True
            fixed_elements = [[]]
            for index in self.covariance.index:
                fixed_elements[0].append(index)

        # If we just take as input a list of str, we just transform it into a list of list
        elif type(fixed_elements) == list and type(fixed_elements[0]) == str:
            fixed_elements = [fixed_elements]

        # Then we fill a list that contnaing the old value of mean
        old_mean_vector = [[]]
        for i in range(len(fixed_elements)):
            for element in fixed_elements[i]:
                old_mean_vector[i].append(mean_in_the_model(element))

        if new_mean_vector == [] or new_mean_vector == [[]]:
            new_mean_vector = old_mean_vector

        elif type(new_mean_vector) == list and type(new_mean_vector[0]) != list:
            new_mean_vector = [new_mean_vector]

        if len(new_mean_vector) != len(fixed_elements):
            raise ValueError(
                f"The total group of the arguments 'fixed_elements' and 'new_mean_vector' isn't matching, {len(fixed_elements)} VS {len(new_mean_vector)} !"
            )

        # dictionnary_r : the variable the we will study the entropy
        # dictionnary_f : the fixed variable

        # Creating a local function to check the type of the arguments and transform them a dict if it not the case
        def list_to_dict(groups):
            if isinstance(groups, list):
                dictionnary = {}
                for i, group in enumerate(groups):
                    dictionnary[str(i)] = group

            elif isinstance(groups, list) == dict:
                dictionnary = groups

            return dictionnary

        dictionary_r = list_to_dict(groups_to_study)
        dictionary_f = list_to_dict(fixed_elements)

        # Transforming the potential list of value in into dictionary to
        if isinstance(old_mean_vector, list):
            old_mean_vector = dict(zip(dictionary_f.keys(), old_mean_vector))
        if isinstance(new_mean_vector, list):
            new_mean_vector = dict(zip(dictionary_f.keys(), new_mean_vector))

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionary_r.keys():
            group = dictionary_r[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(
                        f"The variable '{variable}' in the group argument is not in the covariance matrix !"
                    )

        for key in dictionary_f.keys():
            group = dictionary_f[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(
                        f"The variable '{variable}' in the fixed vector argument is not in the covariance matrix !"
                    )

        # Initialisation of the MI matrix
        entropy = pd.DataFrame(
            index=dictionary_r.keys(), columns=dictionary_f.keys(), dtype=float
        )
        new_Cov = new_mean = dictionary_r

        for key1 in dictionary_r.keys():
            for key2 in dictionary_f.keys():
                # extraction of the list of string
                group_r = dictionary_r[key1]
                group_f = dictionary_f[key2]

                # We create intermediate matrix
                Cov_rr = Cov_df.loc[group_r, group_r].to_numpy(dtype="float64")
                Cov_ff = Cov_df.loc[group_f, group_f].to_numpy(dtype="float64")
                Cov_rf = Cov_df.loc[group_r, group_f].to_numpy(dtype="float64")
                Cov_fr = Cov_rf.T

                # The targeted covariance matrix
                Cov_rr_f = Cov_rr - np.dot(
                    Cov_rf, np.dot(np.linalg.inv(Cov_ff), Cov_fr)
                )
                new_Cov[key1] = Cov_rr_f

                # Return the conditional cov matrix / mean
                entropy.at[key1, key2] = len(Cov_rr) / 2 * np.log(
                    2 * np.pi * np.e
                ) + np.log(np.linalg.det(Cov_rr_f))

                # Mean aspect
                x_f = np.array(new_mean_vector[key2])
                mu_f = np.array(old_mean_vector[key2])
                new_mean[key1] = np.dot(
                    np.dot(Cov_rf, np.linalg.inv(Cov_ff)), (x_f - mu_f)
                )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        if return_Cov_and_mean == True:
            return entropy, new_Cov, new_mean
        else:
            return entropy
    """



    def affine_transormation(self, A : np.ndarray) :
        """
        Fonction to apply an affine transforamtion on the multivariate normal distribution
        y = Ax + b

        A     : Matrix of affine dependency

        """
        cov = self.__covariance

        if A.shape[1] != cov.shape[0] :
            raise IndexError(f"The size of the second dimension of A ({A.shape[1]}) must fit with the size of the first one of the covariance matrix ({cov.shape[0]})")

        return np.dot(A, np.dot(cov, A.T))



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
    ###################   Function plot the MI matrix   #########################
    def plot(self, result="MI", title="", label=False, value_in_cell=False, index_to_keep=[], size_scale=1.0, max_value_display=False):
        """
        Fonction to plot a heatmap of the mutual information

        result     :  specify the data ploted MI/rho/cov

        """

        # Get the dataframe of the result
        result = result.lower()

        if result in ("mi", "mutual information") :
            data_frame = self.MI
        elif result == "rho" or result == "correlation":
            data_frame = self.correlation
        elif result == "cov" or result == "covariance":
            data_frame = self.covariance

        # Look the index to keep for the plot of the matrix
        index_to_keep_bis = []
        # If nothing is specified, we keep everything
        # else
        if index_to_keep != []:
            # We take a look at every index that the user enter
            for index in index_to_keep:
                # If one of them is not in the model, we told him
                if index not in data_frame.index:
                    raise NameError(f"- {index} is not in the correlation matrix")
                # else, we keep in memory the index that are in the model
                else:
                    index_to_keep_bis.append(index)

        else:
            index_to_keep_bis = data_frame.index

        # Then we create a new matrix with only the index specified
        data_frame = data_frame.loc[index_to_keep_bis, index_to_keep_bis]
        matrix = data_frame.to_numpy(dtype="float64")
        
        figesize = (12,6)
        figesize = tuple(x * size_scale for x in figesize)
        fig, ax = plt.subplots(figsize=figesize)

        if result == "mi":

            custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["white", "blue"])

            # Creation of a mask for infinite and 0 values (because its a log scale)
            mask_inf = np.isinf(matrix)
            mask_zero = (matrix == 0)

            # We replace infinite value by NaN values 
            matrix_with_nan = np.where(mask_inf, np.nan, matrix)

            # We replace 0 value by NaN values 
            matrix_with_nan = np.where(mask_zero, np.nan, matrix_with_nan)


            
            # display the matrix with the personalised colormap and LogNorm scale
            condition = np.logical_or(matrix == 0, matrix == np.inf)
            # Vérifie si tous les éléments sont True (c'est-à-dire, soit 0 soit inf)
            all_zero_or_inf = np.all(condition)
            if all_zero_or_inf :
                im = plt.imshow(matrix_with_nan, cmap=custom_map)
            else : 
                epsilon = np.nanmin(matrix_with_nan)
                matrix_with_nan = np.where(np.isnan(matrix_with_nan), epsilon, matrix_with_nan)
                if max_value_display == False :
                    maxi = np.nanmax(matrix_with_nan)
                else :
                    maxi = max_value_display

                im = plt.imshow(matrix_with_nan, cmap=custom_map, norm=matplotlib.colors.LogNorm(vmin=epsilon, vmax=maxi))
            
            # Definitionof the color for the Nan and the values out of bound
            im.cmap.set_bad(color='white')  # 0 in white
            im.cmap.set_over(color='black')  # infinite in black

            # Superposition of the black mask with the infinite values
            matrix_inf_mask = np.ma.masked_where(~mask_inf, mask_inf)
            
            plt.imshow(matrix_inf_mask, cmap=matplotlib.colors.ListedColormap(['black']), alpha=1.0)

            x = np.arange(matrix.shape[1])
            y = np.arange(matrix.shape[0])
            matrix_zero_mask = np.ma.masked_where(~mask_zero, mask_zero)

            # Use `pcolor` to add hatching for zero values
            plt.pcolor(x, y, matrix_zero_mask, hatch='//', alpha=0.0)  # Hatching for 0 values



        elif result == "rho":
            custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["red", "white", "blue"])
            if max_value_display == False :
                max_value_display = 1
            im = plt.imshow(matrix, cmap=custom_map, vmin=-max_value_display, vmax=max_value_display)

        elif result == "cov":
            custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["red", "white", "blue"])
            
            # Set the max of the scale
            if max_value_display == False :
                max_abs = np.max(np.abs(matrix))
            else :
                max_abs = max_value_display

            # Set to 0 the value under the thershold
            threshold = 1e-10
            matrix = np.where(np.abs(matrix) < threshold, 0, matrix)
            
            im = plt.imshow(matrix, cmap='RdBu', vmin=-max_abs, vmax=max_abs)

        # Display the label next to the axis
        if label == True:
            ax.set_xticks(np.arange(len(data_frame.index)), labels=data_frame.index)
            ax.set_yticks(np.arange(len(data_frame.index)), labels=data_frame.index)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10 * size_scale)
            plt.setp(ax.get_yticklabels(), fontsize=10 * size_scale)

        # Display the value of each cell
        if value_in_cell == True:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        round(matrix[i, j], 2),
                        ha="center",
                        va="center",
                        color="black",
                    )

        if title == "":
            if result.lower() == "mi":
                title = "Mutual information"
            elif result.lower() == "rho":
                title = "Correlation"

        # Title of the plot
        ax.set_title(title, fontsize=14 * size_scale)
        fig.tight_layout()

        # Plot of the black line to separate the parameters from the variables
        # Width of the line
        line_width = 1
        # Number of parameters
        N_para = self.parameters.df.shape[0]
        # Position of the line
        x_p_e = [-0.5, N_para - 0.5]
        y_p_e = [N_para - 0.5, N_para - 0.5]
        plt.plot(x_p_e, y_p_e, "black", linewidth=line_width)
        plt.plot(y_p_e, x_p_e, "black", linewidth=line_width)

        x_p = [-0.5, N_para - 0.5]
        y_p = [N_para - 0.5, N_para - 0.5]
        plt.plot(x_p, y_p, "black", linewidth=line_width)
        plt.plot(y_p, x_p, "black", linewidth=line_width)

        plt.colorbar(im)
        plt.show()

        return(fig, ax, im)
    

    ###################################################################################################
    #############   Function to plot the coviarance matrix as the sum of matrices   ######################
    def plot_focus(self, focused, studied="internal", result="MI", label=True) :
        ### Description of the fonction
        """
        Fonction to plot only the line of the studied of the rho or MI 
        
        Parameters
        ----------

        """    
        # Get the dataframe of the result
        result = result.lower()

        if result in ("mi", "mutual information") :
            data_frame = self.MI

        if isinstance(focused, str) :
            focused = [focused]
        
        for focus in focused :
            if focus not in data_frame.index :
                raise NameError(f"The focused input '{focus}' isn't in the dataframe !\n")

        data_frame = data_frame.loc[focused]

        if studied in ("internal") :
            data_frame = data_frame[self.N_without_ext.index] 


        matrix=data_frame.to_numpy()

        index_labels = data_frame.index 
        column_labels = data_frame.columns

        # Paramètres
        n_indexes = matrix.shape[0]
        n_columns = matrix.shape[1]

        bar_width = 0.9/n_indexes  # Largeur de chaque barre
        x = np.arange(n_columns)

        # Création de la figure avec deux subplots : un pour les barres, un pour la heatmap
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Subplot 1 : Graphique en barres groupées
        for i in range(n_indexes):
            ax1.bar(x + 1/(n_indexes+1)*bar_width +(i) * bar_width, matrix[i], bar_width, label=index_labels[i])

        # Paramétrage des labels et des ticks
        ax1.set_xlabel('Studied')
        ax1.set_ylabel('Mutual Information')
        ax1.set_title('')
        ax1.set_xticks(x + bar_width / 2)
        ax1.set_xticklabels(column_labels, rotation=90, ha="center")
        ax1.legend()

        # Subplot 2 : Heatmap des mêmes valeurs en dessous
        heatmap = ax2.imshow(matrix, cmap='Blues', aspect='auto')

        # Affichage des labels sur l'axe des abscisses pour la heatmap
        ax2.set_xticks(np.arange(n_columns))
        ax2.set_xticklabels(column_labels)
        ax2.set_yticks(np.arange(n_indexes))
        ax2.set_yticklabels(index_labels)
        ax2.set_xticklabels(column_labels, rotation=90, ha="center")

        # Display of the colorbar
        cbar = fig.colorbar(heatmap, ax=ax2, orientation='vertical')
        cbar.set_label('Values')

        plt.tight_layout()
        plt.show()


    #############################################################################
    ###################   Function plot the boxplot   ###########################
    def boxplot(self,         
                elements_to_fixe: list,
                elements_to_study=[],
                new_mean_fixed=[],
                title = "Study of fixed variable",
                color_old = "blue",
                color_new = "red"):
        """
        Fonction to plot a boxplot

        """
        # First, we check if the studied variables are all in the model
        # The case where the user enter something
        if elements_to_study != []:
            for name in elements_to_study:
                if name not in self.covariance.index:
                    raise NameError(f"The name variable {name} is not in the model !")


        # Same for the fixed variable
        for fixed_element in elements_to_fixe :
            if fixed_element not in self.covariance.index:
                raise NameError(f"The input fixed variable '{fixed_element}' is not in the model !")
        

        # Then we recover the values that we want to plot by the call of the group_entropy_fixed_vector function
        SD_df, mean_df = self.group_entropy_fixed_vector(elements_to_fixe=elements_to_fixe, elements_to_study=elements_to_study, new_mean_fixed=new_mean_fixed, return_all=True)

        # initialisation of the lists that will contain all the data for the plot
        data_plot = []
        positions_box = []
        positions_label = []
        labels = []
        colors = []

        # For every elements
        for i, element in enumerate(SD_df.index):
            # The old boxplot
            data_plot.append(
                [
                    mean_df.at[element, "Old mean"] + SD_df.at[element, "Old SD"],
                    mean_df.at[element, "Old mean"] - SD_df.at[element, "Old SD"],
                ]
            )
            positions_box.append(2 * i - 0.3)
            positions_label.append(2 * i)
            labels.append(element)
            colors.append(color_old)

            # The new boxplot
            data_plot.append(
                [
                    mean_df.at[element, "New mean"] + SD_df.at[element, "New SD"],
                    mean_df.at[element, "New mean"] - SD_df.at[element, "New SD"],
                ]
            )
            positions_box.append(2 * i + 0.3)
            positions_label.append(2 * i)
            labels.append(" ")
            colors.append(color_new)

        # We plot !!!
        fig, ax = plt.subplots()
        bp = plt.boxplot(
            data_plot,
            positions=positions_box,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            showcaps=False,
            whis=0,
        )

        # Set labels location
        xticks(positions_label)

        # Ereasing of the median
        for median in bp["medians"]:
            median.set_visible(False)

        # Set of the color of the plotbox
        for box, color in zip(bp["boxes"], colors):
            box.set_facecolor(color)

        # Add of the title
        plt.title(title)
        plt.ylabel("Values")

        # Legend
        legend_elements = [
            Patch(facecolor=color_old, edgecolor=color_old, label="Old"),
            Patch(facecolor=color_new, edgecolor=color_new, label="New"),
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        # Rotation of the labels
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )

        # Add of black line between parameter, metabolite and flux
        pos_line = -1
        for element in SD_df.index:
            if self.find(element) == "parameter":
                pos_line += 2
        plt.axvline(x=pos_line, color="black", linestyle="--", linewidth=1)

        for element in SD_df.index:
            if self.find(element) == "metabolite":
                pos_line += 2
        plt.axvline(x=pos_line, color="black", linestyle="--", linewidth=1)

        # Display of the boxplot
        plt.show()



    #############################################################################
    #############   Function display a graphic interface   ######################
    def graphic_interface(self) :
        ### Description of the fonction
        """
        Fonction to display a window with a graphic interface
        
        Parameters
        ----------

        """
        # Fonction to plot boxplot depending of the variable selected by the user
        def update_plot(*args):
            # Get the value of the interactive interface
            fixed_element = combo_box.get()
            fixed_value = slider.get()

            # Delete the previous draw
            plt.clf()
            plt.close('all')
            
            # Deleting of the old graph
            for widget in frame.winfo_children():
                widget.destroy()
                
            # Then we recover the values that we want to plot by the call of the group_entropy_fixed_vector function
            SD_df, mean_df = self.group_entropy_fixed_vector(elements_to_fixe=[fixed_element], elements_to_study=[], new_mean_fixed=[fixed_value], return_all=True)

            # initialisation of the lists that will contain all the data for the plot
            data_plot = []
            positions_box = []
            positions_label = []
            labels = []
            colors = []

            color_old = "blue"
            color_new = "red"

            # For every elements
            for i, element in enumerate(SD_df.index):
                # The old boxplot
                data_plot.append(
                    [
                        mean_df.at[element, "Old mean"] + SD_df.at[element, "Old SD"],
                        mean_df.at[element, "Old mean"] - SD_df.at[element, "Old SD"],
                    ]
                )
                positions_box.append(2 * i - 0.3)
                positions_label.append(2 * i)
                labels.append(element)
                colors.append(color_old)

                # The new boxplot
                data_plot.append(
                    [
                        mean_df.at[element, "New mean"] + SD_df.at[element, "New SD"],
                        mean_df.at[element, "New mean"] - SD_df.at[element, "New SD"],
                    ]
                )
                positions_box.append(2 * i + 0.3)
                positions_label.append(2 * i)
                labels.append(" ")
                colors.append(color_new)

            # We plot !!!
            fig, ax = plt.subplots()
            bp = plt.boxplot(
                data_plot,
                positions=positions_box,
                labels=labels,
                patch_artist=True,
                showfliers=False,
                showcaps=False,
                whis=0,
            )

            # Set labels location
            xticks(positions_label)

            # Ereasing of the median
            for median in bp["medians"]:
                median.set_visible(False)

            # Set of the color of the plotbox
            for box, color in zip(bp["boxes"], colors):
                box.set_facecolor(color)

            # Add of the title
            plt.title("title")
            plt.ylabel("Values")

            # Legend
            legend_elements = [
                Patch(facecolor=color_old, edgecolor=color_old, label="Old"),
                Patch(facecolor=color_new, edgecolor=color_new, label="New"),
            ]
            plt.legend(handles=legend_elements, loc="upper right")

            # Rotation of the labels
            plt.setp(
                ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

            # Add of black line between parameter, metabolite and flux
            pos_line = -1
            for element in SD_df.index:
                if self.find(element) == "parameter":
                    pos_line += 2
            plt.axvline(x=pos_line, color="black", linestyle="--", linewidth=1)

            for element in SD_df.index:
                if self.find(element) == "metabolite":
                    pos_line += 2
            plt.axvline(x=pos_line, color="black", linestyle="--", linewidth=1)

            # Display the graph in the graphique interface
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



        # Creation of the main windows
        root = tk.Tk()
        root.title("Interface Graphique")

        # Creation of the frame for the boxplot
        frame = tk.Frame(root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # COMBOBOX
        data_options = self.covariance.index.to_list()
        selected_data = tk.StringVar()
        combo_box = ttk.Combobox(root, textvariable=selected_data, values=data_options)
        combo_box.pack(side=tk.LEFT, padx=10, pady=10)
        combo_box.current(0)  # default value
        combo_box.bind("<<ComboboxSelected>>", update_plot)  # Lier la fonction update_plot à l'événement de sélection du combobox

        # SLIDER
        slider = tk.Scale(root, from_=0., to=10., orient=tk.HORIZONTAL, length=200, resolution=1, command=update_plot)
        slider.pack(side=tk.LEFT, padx=10, pady=10)
        slider.set(5)  # default value

        # UPDATE BUTTON
        update_button = tk.Button(root, text="Update", command=update_plot)
        update_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Mettre à jour le graphique initial
        update_plot()

        # Lancer la boucle principale
        root.mainloop()


    ###################################################################################################
    #############   Function to plot the coviarance matrix as the sum of matrices   ######################
    def covariance_decomposition(self) :
        ### Description of the fonction
        """
        Fonction to plot the covariance matrix as the sum of covariance matrices that represent each the inflence of 1 parameter
        
        Parameters
        ----------

        """
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

        regroup_type = False
        list_para = self.parameters.df.index

        dict_para = {}
        maxi = mini = 0

        for para in list_para :
            R_para = self.R[[para]].values

            matrix = self.parameters.df.at[para, "Standard deviation"]*np.dot(R_para, R_para.T)

            dict_para[para] = matrix

            if matrix.max() > maxi :
                maxi = matrix.max()
            if matrix.min() < mini :
                mini = matrix.min()

        colors = ['blue', 'white', 'red']
        n_bins = [100, 1, 100]  # Le nombre de bins pour chaque segment
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=sum(n_bins))

        global current_colorbar
        current_colorbar = None
        # Fonction pour afficher la heatmap de la matrice sélectionnée

        def show_heatmap(event):
            global current_colorbar

            selected_key = combobox.get()
            if selected_key in dict_para:
                matrix = dict_para[selected_key]
                ax.clear()
                limit = max(abs(matrix.min()), matrix.max())
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                heatmap = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
                ax.set_title(f"Heatmap de {selected_key}")

                # Supprimer la colorbar existante si elle existe
                if current_colorbar is not None:
                    current_colorbar.remove()
                
                # Ajouter une nouvelle colorbar et mettre à jour la référence
                current_colorbar = fig.colorbar(heatmap, ax=ax)

                canvas.draw()

        # Création de la fenêtre principale
        root = tk.Tk()
        root.title("Heatmap Viewer")

        # Création de la figure et de l'axe pour la heatmap
        fig, ax = plt.subplots(figsize=(6, 4))

        # Intégration de la figure matplotlib dans l'interface tkinter
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Création d'un combobox pour sélectionner la clé du dictionnaire
        combobox = ttk.Combobox(root, values=list(dict_para.keys()))
        combobox.pack(pady=10)
        combobox.bind("<<ComboboxSelected>>", show_heatmap)

        # Démarrage de la boucle principale de l'interface graphique
        root.mainloop()

        return(maxi, mini)

    #############################################################################
    #############   Function to display the escher map  #########################
    def escher_information(self, studied="atp_c", result= "rho", normalized = False, model_json= None, map_json = None):
        ### Description of the fonction
        """
        Fonction display the information shared or correlation between elements on the Escher map
        
        Parameters
        ----------

        studied : str
            Name of the central element\n
        
        result : str
            type of the display result (by default the correlation)\n

        model_json : str
            directory of the json file of the model\n
        
        map_json :
            directory of the json file of the map
        """

        # Setting of the escher map
        import escher
        from escher import Builder

        escher.rc['never_ask_before_quit'] = True

        # If nothing is taken as inpout for the Escher map, we use the default values
        if model_json == None :
            model_json = self.default_JSON
        if map_json == None :
            map_json = self.default_Escher

        # Definition of the Escher Builder
        builder = Builder(
            height=600,
            map_name=None,
            model_json = model_json,
            map_json= map_json,)

        result = result.lower()

        dict_value_meta = {}
        dict_value_flux = {}

        # Display of the mutual information
        if result == "mi" :
            # Definition of the matrix of value
            matrix = self.MI

            # Change of the scale of the circles displayed
            builder.metabolite_scale = [
            { 'type': 'value', 'value': 0.0, 'color': 'rgba(100, 100, 100, 0.0)', 'size': 20},
            { 'type': 'max'  ,               'color': 'rgba(  0,   0, 100, 1.0)', 'size': 40} ]
            
            # Change of the scale of the arrows displayed
            builder.reaction_scale = [
            { 'type': 'value', 'value': 0.0, 'color': 'rgba(  0, 0,   0, 1.0)', 'size':  0},
            { 'type': 'max'  ,               'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

            # For every metabolite of the model (even the external one)
            for meta in self.metabolites.df.index :
                # If the metabolite is internal
                if not self.metabolites.df.at[meta, "External"] :
                    dict_value_meta[meta] = matrix.at[studied, meta]
                else :
                    dict_value_meta[meta] = matrix.at[studied, meta+"_para"]
            
            # For every reaction of the model
            for flux in self.reactions.df.index :
                # We add its value of the intersection of the matrix between the flux and the element
                dict_value_flux[flux] = matrix.at[studied, flux]


        #Display of correlation
        elif result == "rho" or result == "correlation" :

            # Definition of the matrix of value
            matrix = self.correlation
            
            # Change of the scale of the circles displayed
            builder.metabolite_scale = [
            { 'type': 'value', 'value':-1.0, 'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
            { 'type': 'value', 'value': 0.0, 'color': 'rgba(100, 0, 100, 0.0)', 'size':  0},
            { 'type': 'value', 'value': 1.0, 'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]
            
            # Change of the scale of the arrows displayed
            builder.reaction_scale = [
            { 'type': 'value', 'value': 0.0, 'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
            { 'type': 'value', 'value': 1.0, 'color': 'rgba(100, 0, 100, 0.0)', 'size':  0},
            { 'type': 'value', 'value': 2.0, 'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

            # For every metabolite of the model (even the external one)
            for meta in self.metabolites.df.index :
                # If the metabolite is internal
                if not self.metabolites.df.at[meta, "External"] :
                    dict_value_meta[meta] = matrix.at[studied, meta]
                else :
                    dict_value_meta[meta] = matrix.at[studied, meta+"_para"]
            
            # For every reaction of the model
            for flux in self.reactions.df.index :
                # We add its value of the intersection of the matrix between the flux and the element
                dict_value_flux[flux] = matrix.at[studied, flux] + 1

        # Display of response matrix
        elif result == "r" :

            # Definition of the matrix of value
            if normalized == False :
                matrix = self.R
            else :
                matrix = self.R_normalized

            # If the studied variable is in the index => its a flux or a metabolite concentration
            if studied in matrix.index :
                # We look for the interacition of an internal componant with parameters, so a row of R
                matrix = matrix.loc[[studied]].T
                maxi = matrix.abs().max().max()


                builder.metabolite_scale = [
                { 'type': 'value', 'value':-1, 'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
                { 'type': 'value', 'value': 0.0,  'color': 'rgba(100, 0, 100, 0.0)', 'size':  0},
                { 'type': 'value', 'value': 1, 'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

                builder.reaction_scale = [
                { 'type': 'value', 'value':-1, 'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
                { 'type': 'value', 'value': 0.0,  'color': 'rgba(100, 0, 100, 0.0)', 'size':  0},
                { 'type': 'value', 'value': 1, 'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

                # We look only for internal metabolite
                for meta in self.metabolites.df.index :
                    # If the metabolite is external
                    if self.metabolites.df.at[meta, "External"] :
                        dict_value_meta[meta] = matrix.at[meta+"_para", studied]

                # For every enzyme of the model
                for enzyme in self.enzymes.df.index :
                    # We add its value of the intersection of the matrix between the flux and the element
                    dict_value_flux[self.enzymes.df.at[enzyme, "Reactions linked"][0]] = matrix.at[enzyme+"_para", studied]

            # Else, the studied variable is a parameter
            elif studied in matrix.columns :
                # We look for the interacition of a parameter with internal componants, so a coumn of R
                matrix = matrix[[studied]]
                maxi = matrix.abs().max().max()

                if True :
                    for index in matrix.index :
                        for column in matrix.columns :
                            if matrix.at[index, column] > 0 :
                                matrix.at[index, column] = 1
                            else :
                                matrix.at[index, column] = -1

                
                builder.metabolite_scale = [
                { 'type': 'value', 'value':-1, 'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
                { 'type': 'value', 'value': 0.0,  'color': 'rgba(100, 0, 100, 0.0)', 'size':  0},
                { 'type': 'value', 'value': 1, 'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

                builder.reaction_scale = [
                { 'type': 'value', 'value':-1, 'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
                { 'type': 'value', 'value': 0.0,  'color': 'rgba(100, 0, 100, 0.0)', 'size':  0},
                { 'type': 'value', 'value': 1, 'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}]

                # We look only for internal metabolite
                for meta in self.metabolites.df.index :
                    # If the metabolite is internal
                    if not self.metabolites.df.at[meta, "External"] :
                        dict_value_meta[meta] = matrix.at[meta, studied]

                # For every reaction of the model
                for flux in self.reactions.df.index :
                    # We add its value of the intersection of the matrix between the flux and the element
                    dict_value_flux[flux] = matrix.at[flux, studied]


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


    #############################################################################
    #############   Function to display the escher map  #########################
    def escher_mean_deviation(self, fixed_element=[], fixed_value=[], model_json= None, map_json = None):
        ### Description of the fonction
        """
        Fonction display the Escher map of the model with the deviation of the mean after the fixation of a variable as value
        
        Parameters
        ----------

        fixed : str
            Name of the central metabolite\n
        
        fixed_value : str
            type of the display result (by default the correlation)\n

        model_json : str
            directory of the json file of the model\n
        
        map_json :
            directory of the json file of the map
        """
        import escher
        from escher import Builder

        escher.rc['never_ask_before_quit'] = True

        # If nothing is taken as inpout for the Escher map, we use the default values
        if model_json == None :
            model_json = self.default_JSON
        if map_json == None :
            map_json = self.default_Escher

        # Definition of the Escher Builder
        builder = Builder(
            height=600,
            map_name=None,
            model_json = model_json,
            map_json= map_json,
        )
        

        # Scale of the circle
        builder.metabolite_scale = [
        { 'type': 'min'  ,               'color': 'rgba(100, 0,   0, 1.0)', 'size': 40},
        { 'type': 'value', 'value': 0.0, 'color': 'rgba(100, 0, 100, 0.0)', 'size': 20},
        { 'type': 'max'  ,               'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}
        ]

        # mean dataframe
        mean_df = self.group_entropy_fixed_vector(elements_to_fixe=fixed_element, elements_to_study=[], new_mean_fixed=fixed_value, return_all=True)[1]

        dict_value = {}

        # For every metabolite of the model
        for meta in self.metabolites.df.index :
            # If the metabolite is internal
            if not self.metabolites.df.at[meta, "External"] :
                dict_value[meta] = mean_df.at[meta, "Delta mean"]
            else :
                dict_value[meta] = mean_df.at[meta+"_para", "Delta mean"]

        # Implementation of the value to the escher builder
        builder.metabolite_data = dict_value

        # If the metabolite aren't in the model, then they are not display
        builder.metabolite_no_data_size = 5
        builder.metabolite_no_data_color = 'rgba(100, 100, 100, 1.0)'


        from IPython.display import display
        display(builder)

    #############################################################################
    #############   Function to display the escher map  #########################
    def escher_reference(self, model_json= None, map_json = None):
        ### Description of the fonction
        """
        Fonction display the Escher map with the reference value
        
        Parameters
        ----------

        model_json : str
            directory of the json file of the model\n
        
        map_json :
            directory of the json file of the map
        """
        import escher
        from escher import Builder

        escher.rc['never_ask_before_quit'] = True

        # If nothing is taken as inpout for the Escher map, we use the default values
        if model_json == None :
            model_json = self.default_JSON
        if map_json == None :
            map_json = self.default_Escher

        # Definition of the Escher Builder
        builder = Builder(
            height=600,
            map_name=None,
            model_json = model_json,
            map_json= map_json,
        )
        

        dict_value = {}

        # For every metabolite of the model
        for meta in self.metabolites.df.index :
            dict_value[meta] = self.metabolites.df.at[meta, "Concentration"]

        builder.metabolite_scale = [
        { 'type': 'value', 'value': 0.0, 'color': 'rgba(  0, 0, 100, 0.0)', 'size': 10},
        { 'type': 'max'  ,               'color': 'rgba(  0, 0, 100, 1.0)', 'size': 40}
        ]

        builder.reaction_scale = [
        { 'type': 'value', 'value': 0.0, 'color': 'rgba(  0, 0,   0, 1.0)', 'size':  0},
        { 'type': 'max'  ,               'color': 'rgba(  0, 0, 100, 1.0)', 'size': 30}
        ]


        # Implementation of the value to the escher builder
        builder.metabolite_data = dict_value

        dict_value = {}
        for react in self.reactions.df.index :
            dict_value[react] = self.reactions.df.at[react, "Flux"]

        builder.reaction_data = dict_value

        # If the metabolite aren't in the model, then they are not display
        builder.metabolite_no_data_size = 5
        builder.metabolite_no_data_color = 'rgba(100, 100, 100, 1.0)'




        from IPython.display import display
        display(builder)




    #############################################################################
    #############   Function to display the CE and MI   #########################
    def plot_entropy(self, studied:str, reversed = False) :

        index = self.MI.index

        if studied not in index :
            raise NameError(f"The input name '{studied}' isn't in the model !")
        
        #pos_studied = index.get_loc(studied)
        entropy_studied = self.entropy.at[studied, "Entropy"]

        # In this case, we look for the entropy of the other element, and the part of MI shared with the studied variable
        if reversed==False:
            entropy_c = self.entropy_conditional[studied].to_numpy()
        # In this case, we look for the entropy of the studied element and the part of MI with the other elements
        else :
            entropy_c = self.entropy_conditional.loc[studied].copy().to_numpy()

        MI = self.MI.loc[studied].copy().to_numpy()



        # If the MI is infinite, we set the MI to the value of the entropy of the studied variable
        # And the conditional entropy to 0
        for i in range(MI.shape[0]):
            if np.isinf(MI[i]) :
                MI[i] = entropy_studied
                entropy_c[i] = 0

        per_cent = np.zeros(MI.shape)
        for i in range(len(MI)) :
            value = 100.*abs(MI[i]/(entropy_c[i]+MI[i]))


            per_cent[i] = value

        # width of the bars
        width_bar = 0.35

        plt.bar(index, MI, width=width_bar, label='MI', color='blue')

        plt.bar(index, entropy_c, bottom=MI, width=width_bar, label='CE', color="darkorange")

        for i in range(len(index)):
            plt.text(index[i], entropy_c[i]+MI[i], str(round(per_cent[i]))+"%", ha='center', va='bottom')


        max_value = np.max(entropy_c+MI)

        plt.title(f'Shared information with {studied}')
        plt.ylabel('Value of information')
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
        plt.legend()
        plt.ylim(0,max_value*1.2)


    #############################################################################
    #############   Function   #########################
    def plot_entropy_fixed(self, fixed:str) :

        index = self.MI.index

        # First, we check if the inputed fixed element is in the model
        if fixed not in index :
            raise NameError(f"The input name '{fixed}' isn't in the model !")
    

        # Matrix of difference of entropy after the fixation of an element
        delta_entropy = np.abs(self.group_entropy_fixed_vector(elements_to_fixe=fixed)["Delta H"].to_numpy())
        
        # Matrix of mutual inforamtion with the fixed element
        MI = self.MI.loc[fixed].to_numpy()
        for i in range(len(MI)) :
            if np.isinf(MI[i]) :
                MI[i] = 0

        print(MI - delta_entropy)


        
        # width of the bars
        width_bar = 0.35

        # Positions des barres pour les deux ensembles de données
        pos1 = np.arange(len(index))
        pos2 = [pos + width_bar for pos in pos1]

        plt.bar(pos1, MI, width=width_bar, label='MI', color='blue')

        plt.bar(pos2, delta_entropy, width=width_bar, label='Delta H', color="darkorange")


        plt.title(f'difference of entropy after fixation {fixed}')
        plt.ylabel('Value of information')
        plt.xticks(pos1 + width_bar/2, index, rotation=45, ha="right", rotation_mode="anchor")
        plt.legend()

    #############################################################################
    ##########   Function to display the element with the best rho  #############
    def display_best(self, studied:str, N:int):
        ### Description of the fonction
        """
        Fonction to display metabolites that share the most of information with the studied elements
        
        Parameters
        ----------

        studied : str
            String of the studied elements of the model\n
        
        N : int
            Number of element to show of
        """
        list_meta = self.N_without_ext.index
        best = {}

        #For every metabolite
        for element in list_meta :
            # If the number of elements in the dictionnary is lower than what is expected, we add the metabolite by default
            if len(best) < N :
                best[element] = self.rho()[studied][element]
            # Else, we look for the lowest element of the dictionnary that contain the best metabolite and we compare the metabolite with it
            else :    
                key_min = min(best, key=lambda k: abs(best[k]))

                # If the new metabolite is greater than the lowest of the dictionnary, we remove the lowest and add the new one
                if abs(self.rho()["Biomass_Ecoli_core_w_GAM"][element]) > best[key_min] :
                    del best[key_min]
                    best[element] = self.rho()["Biomass_Ecoli_core_w_GAM"][element]


        best = dict(sorted(best.items(), key=lambda item: abs(item[1]), reverse=True))

        keys = list(best.keys())
        values = list(best.values())


        norm = [(x + 1) / 2 for x in values]

        # Color map
        cmap = plt.get_cmap('seismic')

        # Creation of color depending of the normalise value
        colors = [cmap(n) for n in norm]

        # Creation of the histogram
        plt.figure(figsize=(10, 5))
        bars = plt.bar(keys, values, color=colors, alpha=1)

        # Labels and title
        plt.xlabel('Correlations')
        plt.ylabel('Metabolites')
        plt.title(f'Metabolite that have the highest correlation with the {studied}')
        plt.axhline(0, color='grey', linewidth=0.8)  # Ligne horizontale à y=0 pour référence

        # tilt the labels
        plt.xticks(rotation=45, ha='right')

        # display the value in the bar
        for i in range(len(keys)):
            plt.text(keys[i], values[i], f'{values[i]:.2f}', ha='center', va='bottom' if values[i] < 0 else 'top')

        # Display
        plt.show()

    ################################################################################
    #                                                                              #
    # objectif function  #  real data # cost # fitness                             #
    #                                                                              #
    ################################################################################
    #                                                                              #
    #                      MODEL FITTING / ROBUSTNESS & COST                       #
    #                                                                              #
    ################################################################################


        




    ################################################################################
    #                                                                              #
    # linear # SBML # XML # Cobra # JSON # CSV # SBtab # check # driving-forces    #
    #                                                                              #
    ################################################################################
    #                                                                              #
    #                      CREATION & LOAD OF EXISTING MODEL                       #
    #                                                                              #
    ################################################################################

    #############################################################################
    ###############  Function to creat a simple linear network ##################
    def creat_linear(self, n: int, grec=True):
        ### Description of the fonction
        """
        Fonction to create a linear system of n metabolite
        
        Parameters
        ----------
        n : int
          Number of metabolite in the linear network

        """
        if n <= 1:
            raise TypeError("Please enter an integer >= 2 !\n")

        else:
            import string 
            list_alphabet = list(string.ascii_uppercase)
            # reinitialisation of the data
            self.__init__()

            matrix = np.array([[0 for i in range(n - 1)] for k in range(n)])

            for i in range(n):
                for j in range(n - 1):
                    if i == j:
                        matrix[i][j] = -1
                    elif i - 1 == j:
                        matrix[i][j] = 1

            noms_lignes = [list_alphabet[i] for i in range(n)]
            if grec :
                noms_colonnes = [r"$\nu$"+str(i) for i in range(n - 1)]
            else :
                noms_colonnes = [f"v_{i}" for i in range(n - 1)]
            # Attribution of the new stoichiometic matrix

            self.Stoichio_matrix_pd = pd.DataFrame(matrix, index=noms_lignes, columns=noms_colonnes)

            self.metabolites.df.loc[list_alphabet[0], "External"] = True
            self.metabolites.df.loc[list_alphabet[n-1], "External"] = True

            for reaction in self.Stoichio_matrix_pd.columns:
                self.elasticity.p.df.at[reaction, "Temperature"] = 0

            self._update_elasticity()

    #############################################################################
    ###############  Function to creat a simple linear network ##################
    def creat_branch(self, grec=True):
        ### Description of the fonction
        """
        Fonction to create a 2 branch system
        """
        import string 
        list_alphabet = list(string.ascii_uppercase)
        # reinitialisation of the data
        self.__init__()
        
        # 7 meetabolites
        n = 7

        # Matrix of 7 metabolite and 6 reactions
        matrix = np.array([[0 for i in range(n - 1)] for k in range(n)])

        # A
        matrix[0][0] = -1
        # B
        matrix[1][0] = 1
        matrix[1][1] = -1
        # C
        matrix[2][1] = 1
        matrix[2][2] = -1
        matrix[2][4] = -1
        # D
        matrix[3][2] = 1
        matrix[3][3] = -1
        # E
        matrix[4][3] = 1
        # F
        matrix[5][4] = 1
        matrix[5][5] = -1
        # G
        matrix[6][5] = 1


        noms_lignes = [list_alphabet[i] for i in range(n)]
        if grec :
            noms_colonnes = [r"$\nu$"+str(i) for i in range(n - 1)]
        else :
            noms_colonnes = [f"v_{i}" for i in range(n - 1)]
        # Attribution of the new stoichiometic matrix

        self.Stoichio_matrix_pd = pd.DataFrame(matrix, index=noms_lignes, columns=noms_colonnes)

        self.metabolites.df.loc[list_alphabet[0], "External"] = True
        self.metabolites.df.loc[list_alphabet[n-3], "External"] = True
        self.metabolites.df.loc[list_alphabet[n-1], "External"] = True

        for reaction in self.Stoichio_matrix_pd.columns:
            self.elasticity.p.df.at[reaction, "Temperature"] = 0

        self._update_elasticity()


    #############################################################################
    ##################   Function to read a CSV/XLS file  #######################
    def setup(self, file_path:str):
        ### Description of the fonction
        """
        Fonction setup the model with .csv file
        
        Parameters
        ----------
        file     : str 
            The directory of the csv file

        """
        df_options = pd.read_csv(file_path, index_col=0, header=0)

        if df_options.at["add enzyme to all reaction", "value"] == True :
            self.enzymes.add_to_all_reaction()
        if df_options.at["add enzyme in parameters", "value"] == True :
            self.parameters.add_enzymes()
        if df_options.at["add external metabolites in parameters", "value"] == True :
            self.parameters.add_externals()
        if df_options.at["half saturated", "value"] == True :
            self.elasticity.s.half_satured()
        if df_options.at["remove temperture", "value"] == True :
            self.parameters.remove("Temperature")



    #############################################################################
    ##################   Function to read a CSV/XLS file  #######################
    def read_CSV(self, file="../Exemples/XLS/ecoli_core_model.xls"):
        ### Description of the fonction
        """
        Fonction read an Excel file
        
        Parameters
        ----------
        file     : str 
            The directory of the Excel file

        """

        df = pd.read_excel(file)
        N = df.drop(df.columns[0], axis=1)
        N = N.drop(N.index)

        for ligne in df.to_numpy():
            N.loc[ligne[0]] = ligne[1:]

        self.Stoichio_matrix_np = N

        self._update_network()

        for meta in self.metabolites.df.index:
            if meta[-3:] == "(e)":
                self.metabolites.df.at[meta, "External"] = True

    #############################################################################
    ###################   Function to read a SBML file  #########################
    def read_SBML(
        self,
        directory="../Exemples/SBML/",
        file_SBML="E_coli_CCM.xml",
        reference_state_metabolites="reference_state_metabolites.tsv",
        reference_state_c="reference_state_c.tsv",
        reference_state_reactions="reference_state_reactions.tsv",
        reference_state_v="reference_state_v.tsv",
        reference_state_keq="reference_state_keq.tsv",
        ignore_error=True):
        ### Description of the fonction
        """
        Fonction read a SBML file
        
        Parameters
        ----------

        directory     : str 
            The directory of the SBML directory \n

        file_SBML     : str
            name of the .xml file \n

        reference_state_metabolites : str 
            name of the database of metabolite name \n

        reference_state_c           : str 
            name the database of metabolite concentration at reference state \n

        reference_state_reactions   : str
            name of the database of reaction name \n

        reference_state_v           : str 
            name of the database of reaction flux at reference state \n

        reference_state_keq         : str
            name of the database of reaction equibrlium constant \n
            
        ignor_error                 : bool
            to specify if you want to continue th reading process, even if there is an error in the SBML file
        """
        import libsbml

        # Reset of the model
        self.reset

        reader = libsbml.SBMLReader()
        document = reader.readSBML(directory + file_SBML)

        n_error = document.getNumErrors()

        # If the user decided to take care of the error in the model
        if n_error != 0 and ignore_error == False:
            raise ValueError(
                f"There is {n_error} error(s) in your SBML file, please :\n-fix it before to use this function \n-Or put the parameter ignore_error too True"
            )

        else:
            if ignore_error == True:
                print(
                    f"There is {n_error} error(s) in you SBML file, but you decided to ignore it, you little rogue !"
                )
            else:
                print(f"0 error detected in your SBML file")

            model = document.getModel()

            N = pd.DataFrame(dtype=float)

            for reaction in model.reactions:
                N[reaction.getName()] = pd.Series([0] * len(N.index), dtype="float64")

                reactants = reaction.getListOfReactants()
                for reactant in reactants:
                    specie = model.getSpecies(reactant.getSpecies())
                    stoichio = reactant.getStoichiometry()

                    if specie.getName() not in N.index:
                        N.loc[specie.getName()] = pd.Series(
                            [0] * len(N.columns), index=N.columns, dtype="float64"
                        )

                    N.loc[specie.getName(), reaction.getName()] = stoichio

                products = reaction.getListOfProducts()
                for product in products:
                    specie = model.getSpecies(product.getSpecies())
                    stoichio = product.getStoichiometry()

                    if specie.getName() not in N.index:
                        N.loc[specie.getName()] = pd.Series(
                            [0] * len(N.columns), index=N.columns, dtype="float64"
                        )

                    N.loc[specie.getName(), reaction.getName()] = stoichio

                list_species = []
                for specie in model.species:
                    list_species.append(specie.getName())
                for specie in list_species:
                    if specie not in N.index:
                        N.loc[specie] = pd.Series([0] * len(N.columns), index=N.columns, dtype="float64")

                N.fillna(0, inplace=True)

            self.Stoichio_matrix_pd = N

            # Set the metabolite as external
            for specie in model.species:
                if specie.boundary_condition:
                    self.metabolites.df.at[specie.name, "External"] = True

        # now we read the reference state
        def tsv_to_list(file_tsv: str):
            # Work only if the file is a single colomn
            file_tsv = open(file_tsv)
            list_meta_tsv = file_tsv.readlines()
            file_tsv.close()
            list_meta_tsv = [element.rstrip("\n") for element in list_meta_tsv]
            return list_meta_tsv

        import os

        # Reading the metabolites list
        if os.path.exists(directory + reference_state_metabolites):
            list_metabolites = tsv_to_list(directory + reference_state_metabolites)
            # Reading the reference states of the concentrations of the metabolites
            if os.path.exists(directory + reference_state_c):
                list_concentrations = tsv_to_list(directory + reference_state_c)
                for i in range(len(list_metabolites)):
                    if list_metabolites[i] in self.metabolites.df.index:
                        # Attribution of the concentrations to the dataframe
                        self.metabolites.df.at[list_metabolites[i], "Concentration"] = float(list_concentrations[i])
                    else:
                        print(f"Warning : The metabolite {list_metabolites[i]} is not in the SBML file of the metabolic network !")

        # Reading the reactions list
        if os.path.exists(directory + reference_state_reactions):
            list_reactions = tsv_to_list(directory + reference_state_reactions)
            # Reading the reference state of the reaction
            if os.path.exists(directory + reference_state_v):
                list_flux = tsv_to_list(directory + reference_state_v)
                for i in range(len(list_reactions)):
                    if list_reactions[i] in self.reactions.df.index:
                        # Attribution of the flux to the dataframe
                        self.reactions.df.at[list_reactions[i], "Flux"] = float(list_flux[i])
                    else:
                        print(
                            f"Warning : The reaction {list_reactions[i]} is not in the SBML file of the metabolic network !"
                        )

            # reading the referecne state of the equilibrium constant
            if os.path.exists(directory + reference_state_keq):
                list_keq = tsv_to_list(directory + reference_state_keq)
                for i in range(len(list_reactions)):
                    if list_reactions[i] in self.reactions.df.index:
                        # Attributon of the keq to the dataframe
                        self.reactions.df.at[list_reactions[i], "Equilibrium constant"] = float(list_keq[i])



    #############################################################################
    ###################   Function to check the model   #########################
    def BadAss2SBML(self, creat_file_model = True, file_name = "sbml_file.xml"):
        ### Description of the fonction
        """
        Fonction create a SBML file from the current model
        
        Parameters
        ----------

        creat_file_model     : bool 
            Possibility to creat a file with the model in a .XML file 
        """
        import libsbml
        def check(value, message):
            """If 'value' is None, prints an error message constructed using
            'message' and then exits with status code 1.  If 'value' is an integer,
            it assumes it is a libSBML return status code.  If the code value is
            LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
            prints an error message constructed using 'message' along with text from
            libSBML explaining the meaning of the code, and exits with status code 1.
            """
            if value is None:
                raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
            elif type(value) is int:
                if value == libsbml.LIBSBML_OPERATION_SUCCESS:
                    return
                else:
                    err_msg = 'Error encountered trying to ' + message + '.' \
                            + 'LibSBML returned error code ' + str(value) + ': "' \
                            + libsbml.OperationReturnValue_toString(value).strip() + '"'
                    raise SystemExit(err_msg)
            else:
                return

        try:
            document = libsbml.SBMLDocument(3, 1)
        except ValueError:
            raise SystemExit('Could not create SBMLDocument object')
        
        # Create the basic Model object inside the SBMLDocument object.  To
        # produce a model with complete units for the reaction rates, we need
        # to set the 'timeUnits' and 'extentUnits' attributes on Model.  We
        # set 'substanceUnits' too, for good measure, though it's not strictly
        # necessary here because we also set the units for individual species
        # in their definitions.
        
        model = document.createModel()
        
        check(model,                              'create model')
        check(model.setId("Ec_core"),             'set model ID')
        check(model.setName("Ec_core"),           'set model name')
        check(model.setTimeUnits("second"),       'set model-wide time units')
        check(model.setExtentUnits("mole"),       'set model units of extent')
        check(model.setSubstanceUnits('mole'),    'set model substance units')

        # Create a unit definition we will need later.  Note that SBML Unit
        # objects must have all four attributes 'kind', 'exponent', 'scale'
        # and 'multiplier' defined.

        per_second = model.createUnitDefinition()
        check(per_second,                         'create unit definition')
        check(per_second.setId('per_second'),     'set unit definition id')
        unit = per_second.createUnit()
        check(unit,                               'create unit on per_second')
        check(unit.setKind(libsbml.UNIT_KIND_SECOND),     'set unit kind')
        check(unit.setExponent(-1),               'set unit exponent')
        check(unit.setScale(0),                   'set unit scale')
        check(unit.setMultiplier(1),              'set unit multiplier')

        # Create species inside this model, set the required attributes
        # for each species in SBML Level 3 (which are the 'id', 'compartment',
        # 'constant', 'hasOnlySubstanceUnits', and 'boundaryCondition'
        # attributes), and initialize the amount of the species along with the
        # units of the amount.
        dict_meta = {}
        for meta in self.metabolites.df.index :
            # Creation of a species in the model
            s = model.createSpecies()
            check(s,                                                                                'create species ' + meta)
            check(s.setId("ID_"+meta),                                                              'set species '+ meta + ' ID')
            check(s.setName(meta),                                                                  'set species '+ meta + ' name')
            check(s.setInitialAmount(self.metabolites.df.at[meta, "Concentration"]),                'set initial amount for ' + meta)
            check(s.setSubstanceUnits('mole'),                                                      'set substance units for ' + meta)
            check(s.setBoundaryCondition(bool(self.metabolites.df.at[meta, "External"])),           'set "boundaryCondition" on ' + meta)
            dict_meta[meta] = s

        # Create parameters object inside this model, set the required
        # attributes 'id' and 'constant' for a parameter in SBML Level 3, and
        # initialize the parameter with a value along with its units.
        for para in self. parameters.df.index :
            # Creation of parameter in the model
            p = model.createParameter()
            check(p,                                                            'create parameter ' + para)
            check(p.setId(para),                                                'set parameter '+para+' id')
            check(p.setConstant(True),                                          'set parameter '+para+' "constant"')
            check(p.setValue(self.parameters.df.at[para, "Mean values"]),       'set parameter '+para+' value')
            check(p.setUnits('per_second'),                                     'set parameter '+para+' units')
            note = "<body xmlns='http://www.w3.org/1999/xhtml'><p>SD:"+str(self.parameters.df.at[para,"Standard deviation"])+"</p></body>"
            check(p.setNotes(note),                                         'set parameter '+para+' notes')
        
        # Create reactions inside this model, set the reactants and products,
        # and set the reaction rate expression (the SBML "kinetic law").  We
        # set the minimum required attributes for all of these objects.  The
        # units of the reaction rate are determined from the 'timeUnits' and
        # 'extentUnits' attributes on the Model object.
        for react in self.reactions.df.index :
            # Creation of a reaction of the model
            r = model.createReaction()
            dict_stoichio = self.reactions.df.at[react, "Metabolites"]

            check(r,                                                                    'create reaction')
            check(r.setId(react),                                                       'set reaction id')
            check(r.setName(react),                                                     'set reaction name')
            check(r.setReversible(bool(self.reactions.df.at[react, "Reversible"])),     'set reaction reversibility flag')
            check(r.createKineticLaw(),                                                 'set reaction kinetic law')
            
            kinetic_law = r.getKineticLaw()
            list_para = libsbml.ListOfParameters
            # Creation and add of parameter to the reaction
            parameter_lb = kinetic_law.createParameter()
            parameter_lb.setId("LOWER_BOUND")
            if self.reactions.df.at[react, "Reversible"] :
                parameter_lb.setValue(-10000.0)
            else :
                parameter_lb.setValue(0.0)
            parameter_lb.setUnits("mmol_per_gDW_per_hr")

            parameter_ub = kinetic_law.createParameter()
            parameter_ub.setId("UPPER_BOUND")
            parameter_ub.setValue(10000.0)
            parameter_ub.setUnits("mmol_per_gDW_per_hr")

            parameter_flux = kinetic_law.createParameter()
            parameter_flux.setId("FLUX_VALUE")
            parameter_flux.setValue(self.reactions.df.at[react, "Flux"])
            parameter_flux.setUnits("mmol_per_gDW_per_hr")




            for meta in dict_stoichio :
                if dict_stoichio[meta] < 0 :
                    r.addReactant(dict_meta[meta],dict_stoichio[meta])
                else : 
                    r.addProduct(dict_meta[meta],dict_stoichio[meta])
            
        # Creation of a .XML file of the model
        if creat_file_model :
            libsbml.writeSBMLToFile(d=document, filename=file_name)
        
        return(document.getModel())




    #############################################################################
    ###################   Function to check the model   #########################
    def BadAss2Cobra(self, creat_file_model = True, file_name = "cobra_file.json"):
        ### Description of the fonction
        """
        Fonction create a Cobra model from the current BadAss model
        
        Parameters
        ----------

        creat_file_model     : bool 
            Possibility to creat a file with the model in a .JSON file \n

        file_name : str
            Name of the file to creat
        """
        import cobra

        # Initialisation of the usfull variable to build the Cobra model
        # The metabolite
        metabolites = self.metabolites.df.index
        
        # And the reaction
        reactions = []
        for react in self.reactions.df.index :
            reactions.append({})
            reactions[-1]["name"] = react
            reactions[-1]["substrates"] = {}
            reactions[-1]["products"] = {}

            dict_stoichio = self.reactions.df.at[react, "Metabolites"]

            for meta in dict_stoichio.keys() :
                
                if dict_stoichio[meta] < 0 :
                    reactions[-1]["substrates"][meta] = -float(dict_stoichio[meta])
                else : 
                    reactions[-1]["products"][meta] = float(dict_stoichio[meta])


        # Creation of an empty model
        model = cobra.Model()

        # Add of the metabolite to the model
        for metabolite in metabolites:
            model.add_metabolites(cobra.Metabolite(metabolite))

        # Add of the reaction in the model
        for reaction_data in reactions:
            reaction = cobra.Reaction(reaction_data['name'])
            
            # Add of the substrate to the model
            for substrate, coefficient in reaction_data['substrates'].items():
                reaction.add_metabolites({model.metabolites.get_by_id(substrate): -coefficient})
            
            # Add of the product to the model
            for product, coefficient in reaction_data['products'].items():
                reaction.add_metabolites({model.metabolites.get_by_id(product): coefficient})
            
            model.add_reactions([reaction])
        
        if creat_file_model :
            cobra.io.save_json_model(model, filename=file_name)

        return(model)

    #############################################################################
    ###################   Function to read a SBTab file  #########################
    def read_SBtab(
        self,
        filepath = "../Exemples/SBtab/E Coli Core/model.tsv"
    ):
        ### Description of the fonction
        """
        Fonction read a SBTab file
        
        Parameters
        ----------

        directory     : str 
            directory/file_name.tsv
        """

        import sbtab
        import re

        start = time.time()

        # Reset of the model
        self.reset

        # Then we desactivate the automatic update of the model
        self.activate_update = False

        # Function to extract the number from a string
        def extract_name_and_number(string, default = 1):
            # We use a regular expression to find a number followed by a space
            match = re.search(r'(\d+(?:\.\d+)?)\s', string)
            if match:
                # Then we extract the number found
                number = float(match.group(1))
                # We delete it from the string
                string = string.replace(match.group(0), '')
                return number, string
            else:
                # If no number where found, we return a default number
                return default, string


        filename = filepath.split('/')[-1]
        St = sbtab.SBtab.read_csv(filepath=filepath, document_name=filename)


        # We attribute the list
        for table in St.sbtabs :
            if table.table_name == "Reaction" :
                reactions = table.value_rows
                dict_index_reactions = table.columns_dict
            elif table.table_name == "Compound" :
                metabolites = table.value_rows
                dict_index_metabolites = table.columns_dict
            elif table.table_name == "Position" :
                positions = table.value_rows
            elif table.table_name == "MetaboliteConcentration" :
                ref_concentrations = table.value_rows
            elif table.table_name == "Metabolic flux" :
                flux = table.value_rows
                index_flux = table.columns_dict['!Value']
            elif table.table_name == "Reaction Gibbs free Energy" :
                Gibbs_free_energy = table.value_rows
            elif table.table_name == "Equilibrium constant" :
                Equilibrium_const = table.value_rows
                index_const = table.columns_dict['!Value']

        
        # First we add the reactions
        for i, reaction in enumerate(reactions) :
            name = reaction[dict_index_reactions['!ID']]
            # We split the equation str in 2, the first part is the reactants, the second one is the products
            reactants, products = reaction[dict_index_reactions['!ReactionFormula']].split(" <=> ")

           
            dict_meta = {}

            # First we deal with the reactants
            # We split each reactants
            reactants = reactants.split(" + ")
            # Avoid problem in the case of a reaction that only consume
            if reactants[0] != '':
                for reactant in reactants :
                    stoichio, name_meta = extract_name_and_number(reactant, default=1)
                    # Add a negative therme to the stoichio coeff to represent the fact that it is a reactant
                    stoichio *= -1
                    # We remove the '_' before and after the name of the ID of the metabolite
                    if name_meta[0] == "_" :
                        name_meta = name_meta[1:]
                    if name_meta[-1] == "_" :
                        name_meta = name_meta[:-1]

                    dict_meta[name_meta] = stoichio

            # Then we deal with the products
            products = products.split(" + ")
            # Avoid problem in the case of a reaction that only produce
            if products[0] != '':
                for product in products :
                    stoichio, name_meta = extract_name_and_number(product, default=1)
                    
                    # We remove the possible prefactor that indicate that it's a reaction
                    if name.startswith("R_"):
                        name = name[2:]
                    # We remove the '_' before and after the name of the ID of the metabolite
                    if name_meta[0] == "_" :
                        name_meta = name_meta[1:]
                    if name_meta[-1] == "_" :
                        name_meta = name_meta[:-1]

                    dict_meta[name_meta] = stoichio

            reversible = reaction[dict_index_reactions['!IsReversible']] == "True"
            
            try :
                rate = float(flux[i][index_flux])
            except UnboundLocalError:
                rate = 1

            try :
                k_eq = float(Equilibrium_const[i][index_const])
            except UnboundLocalError:
                k_eq = 1
            
            self.reactions.add(name, dict_meta, k_eq=k_eq, reversible=reversible, flux=rate)
        

        # Then we add metabolites
        for meta in metabolites :
            # ID of the metabolite
            # If the complete name is necessary, use '!Name' insteed
            name = meta[dict_index_metabolites['!ID']]


            # We remove the possible prefactor that indicate that it's a metabolite
            if name.startswith("M_"):
                name = name[2:]
            # We remove the '_' before and after the name of the ID of the metabolite
            if name[0] == "_" :
                name = name[1:]
            if name[-1] == "_" :
                name = name[:-1]

            external = meta[dict_index_metabolites['!IsConstant']] == "True"
            concentration = float(meta[dict_index_metabolites['!InitialConcentration']])
            
            self.metabolites.add(name, external, concentration)


        self.activate_update = True
        self._update_elasticity()

            




    #############################################################################
    ###################   Function to check the model   #########################
    @property
    def check(self):
        """
        Function to check the use of elements of the BadAss model
        """
        # Check the reaction
        unused_reactions = []
        for react in self.Stoichio_matrix_np.columns.to_list():
            counter = 0
            for meta in self.Stoichio_matrix_np.index.to_list():
                counter += np.abs(self.Stoichio_matrix_np.loc[meta, react])
            if counter == 0:
                unused_reactions.append(react)

        # Check the metabolite
        unused_metabolites = []
        for meta in self.Stoichio_matrix_np.index.to_list():
            counter = 0
            for react in self.Stoichio_matrix_np.columns.to_list():
                counter += np.abs(self.Stoichio_matrix_np.loc[meta, react])
            if counter == 0:
                unused_metabolites.append(meta)

        print("The following reactions are unused : \n")
        for unused_react in unused_reactions:
            print(f"-{unused_react} \n")

        print("\n \n")
        print("The following metabolites are unused : \n")
        for unused_meta in unused_metabolites:
            print(f"-{unused_meta} \n")

        return (unused_reactions, unused_metabolites)


    #############################################################################
    ##########   Function to check the jacobian of the model   ##################
    @property
    def check_unstable(self) :
        """
        Function that check if the BadAss model is unstable
        """
        eigen_values = np.linalg.eigvals(self.Jacobian.to_numpy(dtype="float64"))

        positif = False
        for value in eigen_values:
            if np.real(value) > 0:
                positif = True
                print(value)

        if positif == True:
            print("The jacobian matrix have positives eigen values, that could lead to an unstable state")
        return eigen_values

    def plot_eigen(self, xlim=(-1.6,0.1)):
        data = self.check_unstable
        # Extraction des parties réelles et imaginaires des nombres complexes
        real_part = [num.real for num in data]
        imag_part = [num.imag for num in data]

        # Tracer les nombres complexes sur le plan imaginaire
        plt.scatter(real_part, imag_part, color='blue', marker='o')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.grid(True)
        plt.xlim(xlim)
        plt.show()


    #############################################################################
    #########   Function to check the driving forces of the model   #############
    @property
    def check_driving_forces(self) -> pd.DataFrame :
        """
        Function that check if the driving force of reaction and the k_eq have the same sign
        """
        dataframe_reaction = pd.DataFrame(
            index=self.reactions.df.index,
            columns=["Driving force", "Flux", "Consistency"],
        )

        vector_k_eq = self.reactions.df["Equilibrium constant"]
        vector_c = self.metabolites.df["Concentration"]

        vector_driving_force = np.log(vector_k_eq) - np.dot(self.Stoichio_matrix_np.T, np.log(vector_c))
        dataframe_reaction["Driving force"] = vector_driving_force
        dataframe_reaction["Flux"] = self.reactions.df["Flux"]
        dataframe_reaction["Consistency"] = False

        for index in dataframe_reaction.index:
            if (
                dataframe_reaction.at[index, "Driving force"]
                * dataframe_reaction.at[index, "Flux"]
                >= 0
            ):
                dataframe_reaction.at[index, "Consistency"] = True

        return dataframe_reaction





    #############################################################################
    #######   function to plot an histogram of a value after a sampling   ####### 
    def plot_histogram(data, bins=10):
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=bins, range=(-1, 1), edgecolor='black', alpha=0.7)  # Retirer `color`
        
        plt.xlabel("Correlation")
        plt.ylabel("Number")
        plt.grid(True)
        plt.show()


    #############################################################################
    ###################   function to save a state   ########################
    def save_state(self):
        """
        function to save the state of a model, use .upload this saved state
        """
        # Use of copy.deepcopy because dataframe are mutable = change also there renferencements
        import copy

        self.__original_atributes = {}

        self.__original_atributes["stoichiometry"] = copy.deepcopy(self.Stoichio_matrix_pd)
        self.__original_atributes["metabolites"] = copy.deepcopy(self.metabolites.df)
        self.__original_atributes["reactions"] = copy.deepcopy(self.reactions.df)
        self.__original_atributes["parameters"] = copy.deepcopy(self.parameters.df)
        self.__original_atributes["elasticities_s"] = copy.deepcopy(self.elasticity.s.df)
        self.__original_atributes["elasticities_p"] = copy.deepcopy(self.elasticity.p.df)
        self.__original_atributes["enzymes"] = copy.deepcopy(self.enzymes.df)

    #############################################################################
    ################   function to upload the saved state   #####################
    def upload_state(self):
        """
        function to upload the saved state by .save_state()
        """
        self.Stoichio_matrix_pd = self.__original_atributes["stoichiometry"]
        self.metabolites.df = self.__original_atributes["metabolites"]
        self.reactions.df = self.__original_atributes["reactions"]
        self.parameters.df = self.__original_atributes["parameters"]
        self.elasticity.s.__df = self.__original_atributes["elasticities_s"]
        self.elasticity.p.__df = self.__original_atributes["elasticities_p"]
        self.enzymes.df = self.__original_atributes["enzymes"]

    #############################################################################
    ###################   Function to reset the model   #########################
    @property
    def reset(self):
        self.__init__()
