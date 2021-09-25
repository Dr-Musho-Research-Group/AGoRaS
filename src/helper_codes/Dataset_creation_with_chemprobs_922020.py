import pickle as pk
import pandas as pd
import numpy as np
# import thermo
# from thermo import chemical as ch 
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as molD
from rdkit.Chem import AddHs
import os, glob
os.chdir('/nfs/home/6/tempker/aae')
from fixing_unbalanced_parenthesis2_v2 import *
from math import isnan
folder = 'smaller_gen_using_new_tech_august'
os.chdir('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/'+folder)

# fp = []
# for file in glob.glob("*.pkl"):
#     f = file
#     fp.append(f)
    
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

dirName = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/'
fp = getListOfFiles(dirName)



def openf(pik):
    try:
        with open(pik, 'rb') as f:
                file = pk.load(f)
                return(file)

    except:
        pass
    
    
file = [openf(x) for x in fp]
file = [x for x in file if x]
Eqns = [x for y in file for x in y]
Eqns = list(set(Eqns))
# print('There are' + (len(Eqns)) + "unique equations")
def Smi2Form(smile):
    """
    This uses RDkit to convert smiles to molecular formula
    """
    try:
        m = Chem.MolFromSmiles(smile)
        m2 = Chem.AddHs(m)
        m3 = Chem.MolToSmiles(m2)
        chem = molD.CalcMolFormula(m2,separateIsotopes = False, abbreviateHIsotopes=False)
        
        # chem = molD.CalcMolFormula(m,separateIsotopes = False, abbreviateHIsotopes=True)
    except:
        chem = np.nan
        m3 = np.nan
        # chem = 'bad'

    return chem, m3

delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
               '##', '..', '==', '# ', '((', '))', '++', ' - ','--','-=-']
Eqns_clean = [x for x in Eqns if all(i not in x for i in delete_list)]


eqns = pd.Series(Eqns_clean)
eqns = eqns.replace(" ","", regex = True)


Eqns_split = eqns.str.split('>', n=1, expand = True)
Eqns_split = Eqns_split.dropna()


Prd = Eqns_split[1]
Rxn = Eqns_split[0]

Prd = Prd.str.split('~', expand = True)
Rxn = Rxn.str.split('~', expand = True)
Prd = Prd.replace('',np.nan,regex=True)
Rxn = Rxn.replace('',np.nan,regex=True)


Prd_smiles = Prd.copy()
Rxn_smiles= Rxn.copy()
EQTable = pd.concat([Rxn, Prd], axis=1)



FQTable = EQTable.stack().value_counts()
FQTable = pd.Series(FQTable.index, name='Smiles_Name').str.replace(' ','').tolist()
Name_List = FQTable


FQTable = [checkbalanceleft(x) for x in FQTable]
FQTable = pd.Series([checkbalanceright(x) for x in FQTable])



Form_List = pd.Series([Smi2Form(x) for x in FQTable])

Form_List = Form_List.tolist()
Form_Lists = list(map(list, zip(*Form_List)))
Form_List = Form_Lists[0]
smiles_list = Form_Lists[1]

rep_dict_species = {Name_List[i]: Form_List[i] for i in range(len(Name_List))}
rep_dict_smiles = {Name_List[i]: smiles_list[i] for i in range(len(Name_List))}


def sub_out(df, dictonary):
    try:
        df.iloc[:,0] = df.iloc[:,0].map(dictonary)
    
        df.iloc[:,1] = df.iloc[:,1].map(dictonary)
    
        df.iloc[:,2] =  df.iloc[:,2].map(dictonary)
    
        df.iloc[:,3] =  df.iloc[:,3].map(dictonary)
        
        df.iloc[:,4] =  df.iloc[:,4].map(dictonary)

        
    except:
        pass
    
    return(df)
        
Prod = sub_out(Prd, rep_dict_species)
React = sub_out(Rxn, rep_dict_species)


P_drop = Prod[0].notna()
R_drop = React[0].notna()
Prod = Prod[P_drop]
Prd_smiles = Prd_smiles[P_drop]
React = React[R_drop]
Rxn_smiles  = Rxn_smiles[R_drop]

Rxn_smiles = sub_out(Rxn_smiles, rep_dict_smiles)
Prd_smiles = sub_out(Prd_smiles, rep_dict_smiles)




eqs_species = pd.merge(left=React, left_index = True, right = Prod, right_index = True, how = 'inner')
eqs_species = eqs_species.replace(np.nan, '')
eqs_species = eqs_species.replace(' ', '')
Products_species = eqs_species.filter(regex='_x$',axis=1)
Reactants_species = eqs_species.filter(regex='_y$',axis=1)



eqs_smiles = pd.merge(left=Rxn_smiles, left_index = True, right = Prd_smiles, right_index = True, how = 'inner')
eqs_smiles = eqs_smiles.replace(np.nan, '')
eqs_smiles = eqs_smiles.replace(' ', '')
Products_smiles = eqs_smiles.filter(regex='_x$',axis=1)
Reactants_smiles = eqs_smiles.filter(regex='_y$',axis=1)


eqs_smiles = dict(Reactants_smiles = Reactants_smiles,
           Products_smiles = Products_smiles)

eqs_species = dict(Reactants_species = Reactants_species,
           Products_species = Products_species)

output = open('/nfs/home/6/tempker/aae/transformed_data_chem_eq/eqs_with_smile_copies/eqs_species_alleqs_asof_9820.pkl','wb')
pk.dump(eqs_species, output)

output = open('/nfs/home/6/tempker/aae/transformed_data_chem_eq/eqs_with_smile_copies/eqs_smiles_alleqs_asof_9820.pkl','wb')
pk.dump(eqs_smiles, output)

