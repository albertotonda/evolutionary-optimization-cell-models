#####################
# Library
#####################
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import MODEL

#####################
# Class Regulation
#####################
class Regulation_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance:"MODEL"):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.df = pd.DataFrame(
            columns=[
                "Regulated flux",
                "Regulator",
                "Coefficient of regulation",
                "Type regulation",
                "Activated"
            ]
        )

    #################################################################################
    #########           Return the Dataframe of the            ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return len(self.df.shape)

    #################################################################################
    #########           Fonction to add a regulation                       ##########
    def add(
        self, name: str, regulated: str, regulator: str, coefficient=0.5, allosteric=True, activated = True, **kwargs):
        ### Description of the fonction
        """
        Fonction to add a regulation to the model
        
        Parameters
        ----------

        regulated       : str
            Name of regulated flux\n

        regulator       : str
            Name of the metabolite that regulate\n

        coefficient     : foat
            Coefficient of regulation, coef > 0 => activation, coef < 0 => inihibition\n

        allosteric      : str
            Specify the type of reaction, True => allosteric, False => transcriptional

        activated       : bool
            Is the regualation arrow activated ? 
        """
        if allosteric == True:
            type_regulation = "allosteric"
        else:
            type_regulation = "transcriptional"

        # Look if the Regulation Class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(
                columns=[
                    "Regulated flux",
                    "Regulator",
                    "Coefficient of regulation",
                    "Type regulation",
                    "Activated"
                ]
            )

        # Look if the regulation is already in the regulation dataframe
        elif name in self.df.index:
            raise NameError(
                f"The name of the regulation '{name}' is already in the regulation dataframe !"
            )

        # Look if the regulated flux is in the model
        elif regulated not in self.__class_MODEL_instance.reactions.df.index:
            raise NameError(
                f'The reaction "{regulated}" is not in the reaction dataframe !'
            )

        # We look if the regulator is in the metabolites dataframe
        if regulator not in self.__class_MODEL_instance.metabolites.df.index :
            raise NameError(
                f'The metabolite "{regulator}" is not in the metabolite dataframe !')

        # We look if the inputs are in the right type
        if not isinstance(coefficient, (int,float)) :
            raise TypeError(f"The input argument 'coefficient' must be a float, not a {type(coefficient)} !\n")
        if not isinstance(allosteric, bool) :
            raise TypeError(f"The input argument 'allosteric' must be a bool, not a {type(allosteric)} !\n")
        if not isinstance(activated, bool) :
            raise TypeError(f"The input argument 'activated' must be a bool, not a {type(activated)} !\n")

        # Else it's allright :D
        self.df.loc[name] = [regulated, regulator, coefficient, type_regulation, activated]

        # If the regulation arrow is activated, we modifiy the elasticity of the model
        if activated == True :

            # If the regulation is allosteric
            if allosteric == True :
                # If the regulator is an internal metabolite
                if self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == False:
                    self.__class_MODEL_instance.elasticity.s.regulation.at[
                        regulated, regulator
                    ] += float(coefficient)
                
                # If the regulator is an external metabolite
                elif self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == True:
                    self.__class_MODEL_instance.elasticity.p.df.at[
                        regulated, regulator+"_para"
                    ] += coefficient

            # Else, it is a transcriptionnal regulation
            else:
                # name of the enzyme linked to this regulation
                enzyme = "enzyme_" + name

                # We concidere now this enzyme as a metabolite
                self.__class_MODEL_instance.metabolites.add(name=enzyme)
                self.__class_MODEL_instance.reactions.add(
                    name="creation_" + name, metabolites={enzyme: 1}
                )
                self.__class_MODEL_instance.reactions.add(
                    name="destruction_" + name, metabolites={enzyme: -1}
                )
                if self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == False:
                    self.__class_MODEL_instance.elasticity.s.regulation.at[
                        "creation_" + name, regulator
                    ] += coefficient
                elif self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == True:
                    self.__class_MODEL_instance.elasticity.p.df.at[
                        "creation_" + name, regulator
                    ] += coefficient

                default_name = "para_trans_" + name
                name_new_para = kwargs.get('name_parameter', default_name)

                self.__class_MODEL_instance.parameters.add(name=name_new_para)

            self.__class_MODEL_instance._update_elasticity()


    ######################################################################################################
    #########           Fonction to add differnet regulation arrows from vectors                ##########
    def add_from_vector(
        self, arrow_labels, coefficients:list, signs:list, activated:list, **kwargs):
        ### Description of the fonction
        """
        Fonction to add a regulation to the model
        
        Parameters
        ----------

        arrow_labels     : list
            Vector of list of 2 elements, the regulated and the regulator\n

        coefficients     : list
            Vector of float the represent the intensity of the regulation\n

        signs      : list
            Vector of 1 or -1, the represent if the regulation is an activation or a inibition\n

        activated       : bool
            Vector of Bool that say if the associated arrow is ON or OFF\n
        """
        # We adjuste the magnitude of the arrow depending of the maximum and minimum
        coefficients = (np.array(self.__class_MODEL_instance.MOO.vectors["max"]-self.__class_MODEL_instance.MOO.vectors["min"])
                    )*np.array(coefficients) + self.__class_MODEL_instance.MOO.vectors["min"]
        
        # Filling of the DataFrame
        for label, coef, sign, act in zip(arrow_labels, coefficients, signs, activated):
            regulated_flux, regulator = label

            coefficient_of_regulation = coef * sign
            regulation_type = "activation" if sign >= 0 else "inhibition"
            index = f"{regulated_flux}&{regulator}"
            
            self.df.loc[index] = {
                "Regulated flux": regulated_flux,
                "Regulator": regulator,
                "Coefficient of regulation": coefficient_of_regulation,
                "Type regulation": regulation_type,
                "Activated": act
            }

            if act:
                # If the regulator is an internal metabolite
                if self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == False:
                    self.__class_MODEL_instance.elasticity.s.regulation.at[
                        regulated_flux, regulator
                    ] += coefficient_of_regulation
                
                # If the regulator is an external metabolite
                elif self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == True:
                    self.__class_MODEL_instance.elasticity.p.df.at[
                        regulated_flux, regulator+"_para"
                    ] += coefficient_of_regulation


            self.__class_MODEL_instance._update_elasticity()



    #################################################################################
    #########        Fonction to remove a regulation arrow                ###########
    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove regulation arrow
        
        Parameters
        ----------

        name        : str
            Name of the regulation arrow to remove
        """

        self.inactivate(name)

        self.df.drop(name)



    #################################################################################
    #########           Fonction to change a regulation coefficient        ##########
    def change_coeff(self, name_regu: str, new_coeff: float) -> None:
        ### Description of the fonction
        """
        Fonction to change the coefficient of a regulation effect
        
        Parameters
        ----------

        name_regu       : str
            Name of regulation effect to change\n

        new_coeff       : float
            New value of the regulation coefficient

        """
        # Check if the regulation name is in the dataframe
        if name_regu not in self.df.index:
            raise NameError(
                f"The regulation name '{name_regu}' is not in the regulation dataframe"
            )
        

        else:
            regulated = self.df.at[name_regu, "Regulated flux"]
            regulator = self.df.at[name_regu, "Regulator"]

            # If the regulator is an internal metabolite
            if self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == False:
                
                # if it is an allosteric/direct regulation
                if self.df.at[name_regu, "Type regulation"] == "allosteric":
                    # Soustraction of the old value and addition of the new one on the E_s matrix
                    self.__class_MODEL_instance.elasticity.s.regulation.at[
                        regulated, regulator
                    ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

                # Case of a transcriptional/undirect regulation
                else:
                    # Soustraction of the old value and addition of the new one on the E_s matrix
                    self.__class_MODEL_instance.elasticity.s.regulation.at[
                        "creation_" + name_regu, regulator
                    ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

            # If the regulator is an external metabolite
            elif self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == True:
                # if it is an allosteric/direct regulation
                if self.df.at[name_regu, "Type regulation"] == "allosteric":
                    # Soustraction of the old value and addition of the new one on the E_s matrix
                    self.__class_MODEL_instance.elasticity.p.df.at[
                        regulated, regulator
                    ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

                # Case of a transcriptional/undirect regulation
                else:
                    # Soustraction of the old value and addition of the new one on the E_s matrix
                    self.__class_MODEL_instance.elasticity.p.df.at[
                        "creation_" + name_regu, regulator
                    ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])


            # Attribution of the new value to the coeff
            self.df.at[name_regu, "Coefficient of regulation"] = new_coeff



    #################################################################################
    #########        Fonction to activate a regulation arrow               ##########

    def activate(self, name_regu: str) -> None:
        ### Description of the fonction
        """
        Fonction to activate a regulation arrow
        
        Parameters
        ----------

        name_regu        : str
            Name of the regulation name to activate
        """

        # Look if the regulation is in the model
        if name_regu not in self.df.index:
            raise NameError(
                f"The regulation {name_regu} is not in the regulation dataframe, please enter a valide name \n"
            )
        
        regulated = self.df.at[name_regu, "Regulated flux"]
        regulator = self.df.at[name_regu, "Regulator"]
        coeff = self.df.at[name_regu, "Coefficient of regulation"]

        # If the regulation arrows is initialy inactivated, we modifiy the elasticity
        if self.df.at[name_regu, "Activated"] == False : 
            
            # If the regulator is an internal metabolite
            if self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == False:
                
                # Case where it is an alosteric regulation
                if self.df.at[name_regu, "Type regulation"] == "allosteric" :
                    self.__class_MODEL_instance.elasticity.s.regulation.at[regulated, regulator] += coeff
                
                # Case where itis a transcriptionnal regulation
                elif self.df.at[name_regu, "Type regulation"] == "transcriptional" :
                # name of the enzyme linked to this regulation
                    enzyme = "enzyme_" + name_regu

                    # We concidere now this enzyme as a metabolite
                    self.__class_MODEL_instance.metabolites.add(name=enzyme)
                    # And add 2 reactions of production and degradation
                    self.__class_MODEL_instance.reactions.add(name="production_" + name_regu, metabolites={enzyme: 1})
                    self.__class_MODEL_instance.reactions.add(name="degradation_" + name_regu, metabolites={enzyme: -1})
                    # And then change the coefficient of the elasticity matrix
                    self.__class_MODEL_instance.elasticity.s.regulation.at["production_" + name_regu, regulator] += coeff
        

            # If the regulator is an external metabolite
            elif self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == True:
                
                # Case where it is an alosteric regulation
                if self.df.at[name_regu, "Type regulation"] == "allosteric" :
                    self.__class_MODEL_instance.elasticity.p.df.at[regulated, regulator] += coeff
                
                # Case where itis a transcriptionnal regulation
                elif self.df.at[name_regu, "Type regulation"] == "transcriptional" :
                # name of the enzyme linked to this regulation
                    enzyme = "enzyme_" + name_regu

                    # We concidere now this enzyme as a metabolite
                    self.__class_MODEL_instance.metabolites.add(name=enzyme)
                    # And add 2 reactions of production and degradation
                    self.__class_MODEL_instance.reactions.add(name="production_" + name_regu, metabolites={enzyme: 1})
                    self.__class_MODEL_instance.reactions.add(name="degradation_" + name_regu, metabolites={enzyme: -1})
                    # And then change the coefficient of the elasticity matrix
                    self.__class_MODEL_instance.elasticity.p.df.at["production_" + name_regu, regulator] += coeff


        self.df.at[name_regu, "Activated"] = True

        # Then we update the rest of the model
        self.__class_MODEL_instance._update_network()




    #################################################################################
    #########        Fonction to inactivate a regulation arrow            ###########

    def inactivate(self, name_regu: str) -> None:
        ### Description of the fonction
        """
        Fonction to inactivate a regulation arrow
        
        Parameters
        ----------

        name_regu        : str
            Name of the regulation name to inactivate
        """

        # Look if the regulation is in the model
        if name_regu not in self.df.index:
            raise NameError(
                f"The regulation {name_regu} is not in the regulation dataframe, please enter a valide name \n"
            )
        
        regulated = self.df.at[name_regu, "Regulated flux"]
        regulator = self.df.at[name_regu, "Regulator"]
        coeff = self.df.at[name_regu, "Coefficient of regulation"]

        # If the regulation arrows is initialy activated, we modifiy the elasticity
        if self.df.at[name_regu, "Activated"] == True : 
            
            # If the regulator is an internal metabolite
            if self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == False:

                # Case where it is an alosteric regulation
                if self.df.at[name_regu, "Type regulation"] == "allosteric" :
                    # We substract the coeff
                    self.__class_MODEL_instance.elasticity.s.regulation.at[regulated, regulator] -= coeff
                
                # Case where itis a transcriptionnal regulation
                elif self.df.at[name_regu, "Type regulation"] == "transcriptional" :
                    # name of the enzyme linked to this regulation
                    enzyme = "enzyme_" + name_regu

                    # We remove the enzyme that where considere like a metabolite
                    self.__class_MODEL_instance.metabolites.remove(enzyme)
                    # And its associated reactions
                    self.__class_MODEL_instance.reactions.remove(name="creation_" + name_regu)
                    self.__class_MODEL_instance.reactions.remove(name="destruction_" + name_regu)

                        # If the regulator is an internal metabolite
            elif self.__class_MODEL_instance.metabolites.df.at[regulator, "External"] == True:

                # Case where it is an alosteric regulation
                if self.df.at[name_regu, "Type regulation"] == "allosteric" :
                    # We substract the coeff
                    self.__class_MODEL_instance.elasticity.p.df.at[regulated, regulator] -= coeff
                
                # Case where itis a transcriptionnal regulation
                elif self.df.at[name_regu, "Type regulation"] == "transcriptional" :
                    # name of the enzyme linked to this regulation
                    enzyme = "enzyme_" + name_regu

                    # We remove the enzyme that where considere like a metabolite
                    self.__class_MODEL_instance.metabolites.remove(enzyme)
                    # And its associated reactions
                    self.__class_MODEL_instance.reactions.remove(name="creation_" + name_regu)
                    self.__class_MODEL_instance.reactions.remove(name="destruction_" + name_regu)

        self.df.at[name_regu, "Activated"] = False

        # Then we update the rest of the model
        self.__class_MODEL_instance._update_network()


    #################################################################################
    #########      Fonction to read a file of regulation database          ##########

    def read_file(self, file_path: str) -> None:
        ### Description of the fonction
        """
        Fonction to read a file of regulation database
        
        Parameters
        ----------

        name        : str
            File_path of the regulation database (SBTab format)
        """
        import sbtab

        filename = file_path.split('/')[-1]
        St = sbtab.SBtab.read_csv(filepath=file_path, document_name=filename)

        table = St.sbtabs[0] 

        for reg in table.value_rows :
            
            meta = reg[0]
            flux = reg[1]
            type_reg = reg[2]
            print(f"{meta} ; {flux} ; {type_reg}\n")
            allosteric_list = ["direct", "alosteric"]
            if type_reg.split(" ")[0].lower() in allosteric_list  :
                allosteric = True
            else :
                allosteric = False

            coeff = 1
            if type_reg.split(" ")[1] == "inhibition" :
                    coeff = -1

            self.__class_MODEL_instance.regulations.add(name = f"{type_reg} {meta} -> {flux}", regulated=flux, regulator=meta, coefficient=coeff,allosteric=allosteric)
