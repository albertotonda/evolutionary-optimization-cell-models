#####################
# Library
#####################
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL


#####################
# Class Enzymes
#####################
class Enzymes_class:
    #############################################################################
    ###############             Initialisation              #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.df = pd.DataFrame(columns=["Concentration / Activity", "Reactions linked", "Type"])

    #################################################################################
    ###########           Return the Dataframe of the enzymes           #############
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of enzymes              ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    #########            Fonction to add an enzyme                         ##########
    def add(self, name="", mean=1, reaction_linked=[], Type="parameter") -> None:
        ### Description of the fonction
        """
        Fonction to add an enzyme to the model

        Parameters
        ----------

        name                : str
            Name of the enzyme\n

        mean                : float
            Mean value of the enzyme\n

        reaction_linked     : list of str
            list of reaction names linked to this enzyme

        Type                : str
            Type of the enzyme: parameter mean its a source of uncertainty / variable mean it is an internal component
        """

        # Look if the enzyme is already in the model
        if name not in self.df.index.to_list():
            self.df.loc[name] = [mean, reaction_linked, Type]

    #################################################################################
    #########           Fonction to remove an enzyme                       ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove an enzyme to the model

        Parameters
        ----------

        name        : str
            Name of the enzyme to remove
        """

        # Look if the enzyme is in the model
        if name not in self.df.index:
            raise NameError(
                f"The enzyme {name} is not in the enzyme dataframe, please enter a valide name \n"
            )

        else:
            # Else, the enzyme is remove from the dataframe
            self.df.drop(name, inplace=True)

            # Removing this enzyme from the elasticity matrix E_p
            if name in self.__class_MODEL_instance.elasticity.p.df.index:
                self.__class_MODEL_instance.elasticity.p.df.drop(
                    name, axis=1, inplace=True
                )
                self.__class_MODEL_instance._reset_value()

            # Removing every mention of the enzyme in the operon dataframe
            for operon in self.__class_MODEL_instance.operons.df.index :
                if name in self.__class_MODEL_instance.operons.df.at[operon, "Enzymes linked"] :
                    liste_enzym = [x for x in self.__class_MODEL_instance.operons.df.at[operon, "Enzymes linked"] if x != name]
                    self.__class_MODEL_instance.operons.df.at[operon, "Enzymes linked"] = liste_enzym
                    

    #################################################################################
    #########      Fonction to add an enzyme link to every reaction        ##########
    def add_to_all_reaction(self) -> None:
        ### Description of the fonction
        """
        Fonction to add an parameter-enzyme to every reaction of the model
        """
        for reaction in self.__class_MODEL_instance.reactions.df.index:
            name_enzyme = "enzyme_" + reaction
            # Look if the enzyme is already in the model
            if name_enzyme not in self.df.index.to_list():
                self.add(name_enzyme, 1, [reaction], "parameter")
