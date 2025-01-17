#####################
# Library
#####################
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL

#####################
# Class Parameters
#####################
class Parameter_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.df = pd.DataFrame(
            index=["Temperature"], columns=["Mean values", "Standard deviation"]
        )
        self.df.loc["Temperature", "Mean values"] = 273.15
        self.df.loc["Temperature", "Standard deviation"] = 1.0

    #################################################################################
    #########           Return the Dataframe of the parameters             ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of parameters           ##########
    @property
    def len(self):
        return len(self.df)
    
    #################################################################################
    #########         Fonction to return a list of the parameters          ##########
    @property
    def list(self):
        return(list(self.df.index))

    #################################################################################
    #########           Fonction to add a parameters                       ##########
    def add(self, name: str, mean=1, Standard_deviation=1.0) -> None:
        ### Description of the fonction
        """
        Fonction to add a parameter to the model\n
            If it is already in the model, change the properties


        Parameters
        ----------

        name                : str
            Name of the parameter\n

        mean                : float
            Mean value of the parameter\n

        Standard_deviation  : float
            Standard deviation of the parameter

        """

        # Look if the parameter is already in the model
        if name in self.df.index.to_list():
            raise NameError('The parameter "' + name + '" is already in the model !')

        # Else, the parameter is add to the model by an add to the DataFrame
        else:
            self.df.loc[name] = [mean, Standard_deviation]
            self.__class_MODEL_instance._reset_value("E_p")

    #################################################################################
    #########           Fonction to change a parameter                     ##########

    def change(self, name: str, mean=None, SD=None):
        ### Description of the fonction
        """
        Fonction to change a parameter properties in the model
        
        Parameters
        ----------
        name    : str
            Name of the parameter to change\n
        
        mean    : float
            Mean value of the parameter\n

        SD      : float
            Standard deviation of the parameter uncertainty

        """

        if name not in self.df.index:
            raise NameError(f"The name '{name}' is not in the parameter dataframe")

        else:
            if mean == None :
                mean = self.df.at[name, "Mean values"]
            if SD == None :
                SD = self.df.at[name, "Standard deviation"]

            self.df.at[name, "Mean values"] = mean
            self.df.at[name, "Standard deviation"] = SD


            self.__class_MODEL_instance._reset_value(session="var")


    ##################################################################################
    #########           Fonction to remove a parameter                      ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove a parameter to the model
        
        Parameters
        ----------
        
        name        : str
            Name of the parameter to remove
        """

        # Look if the parameter is in the model
        if name not in self.df.index.to_list():
            raise NameError("Please enter a name of a parameter of the model \n")

        else:
            # Else, the parameter is remove from the dataframe
            self.df.drop(name, inplace=True)

            print(f"Name of the removed parameter : {name}")

            # Removing this parameter from the elasticity matrix E_p
            if name in self.__class_MODEL_instance.elasticity.p.df.columns:
                self.__class_MODEL_instance.elasticity.p.df.drop(
                    name, axis=1, inplace=True
                )
                self.__class_MODEL_instance._reset_value("E_p")

    ##################################################################################
    #########         Fonction to add all enzyme to the model               ##########
    def add_enzymes(self) -> None:
        ### Description of the fonction
        """
        Fonction to consider all enzymes as parameters
        """

        # For every enzymes of the models
        for enzyme in self.__class_MODEL_instance.enzymes.df.index:
            # if this one is not already considered as a parameter
            if (enzyme + "_para") not in self.df.index:
                # We add it in the parameter
                self.add(
                    enzyme + "_para",
                    mean=self.__class_MODEL_instance.enzymes.df.loc[
                        enzyme, "Concentration / Activity"
                    ],
                )
                # We add a new column of 0 to the parameters elasticity dataframe
                self.__class_MODEL_instance.elasticity.p.df[enzyme + "_para"] = 0.0

        # For every reaction of the N matrix
        for reaction in self.__class_MODEL_instance.Stoichio_matrix_pd.columns:
            # if the reaction is not in the Dataframe, we add it
            if reaction not in self.__class_MODEL_instance.elasticity.p.df.index:
                self.__class_MODEL_instance.elasticity.p.df.loc[reaction] = 0.0

        # For every enzymes of the models
        for enzyme in self.__class_MODEL_instance.enzymes.df.index:
            # We add 1 to the enzyme linked to reaction
            for reaction in self.__class_MODEL_instance.enzymes.df.loc[
                enzyme, "Reactions linked"
            ]:
                self.__class_MODEL_instance.elasticity.p.df.loc[
                    reaction, enzyme + "_para"
                ] = 1.0

        self.__class_MODEL_instance._reset_value("E_p")

    ##################################################################################
    #########         Fonction to add all external metabolite               ##########
    def add_externals(self) -> None:
        ### Description of the fonction
        """
        Fonction to consider all external metabolite as parameters
        """


        # For every metabolite of the model
        missing_meta_ext = []
        for meta in self.__class_MODEL_instance.metabolites.df.index:
            # If this metabolite is external
            if self.__class_MODEL_instance.metabolites.df.at[meta, "External"] == True:
                # If this external metabolite is'nt already in the parameter dataframe
                if (meta + "_para") not in self.df.index:
                    # We add it to the parameter dataframe
                    self.add(meta + "_para", mean = self.__class_MODEL_instance.metabolites.df.at[meta, "Concentration"])

                # Same for the parameters in the elasticity matrix
                if (
                    meta + "_para"
                ) not in self.__class_MODEL_instance.elasticity.p.df.columns:
                    missing_meta_ext.append(meta)
                    # self.__class_MODEL_instance.elasticity.p.df[meta + "_para"] = 0.0

        # Then we add every columns at the elasticity matrix
        if missing_meta_ext != []:
            # By creating a temporary dataframe with the external metabolite as columns
            df_temporary = -0.5 * self.__class_MODEL_instance.N.loc[missing_meta_ext]

            for index in df_temporary.index :
                df_temporary = df_temporary.rename(index={index: index+"_para"})

            df_temporary = df_temporary.T
        
            self.__class_MODEL_instance.elasticity.p.df = pd.concat(
                [self.__class_MODEL_instance.elasticity.p.df, df_temporary], axis=1
            )

        # Creation of a temporary dataframe with just the metabolite
        df_temporary = pd.DataFrame(
            index=self.__class_MODEL_instance.elasticity.p.df.index
        )

        self.__class_MODEL_instance.elasticity.p.df = pd.concat(
            [self.__class_MODEL_instance.elasticity.p.df, df_temporary], axis=1
        )

        self.__class_MODEL_instance._reset_value("E_p")
