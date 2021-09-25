#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:13:41 2020

@author: tempker
"""

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
folder = 'both'
os.chdir('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/'+folder)


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
        m.UpdatePropertyCache(strict=False)
        m2 = Chem.AddHs(m)
        chem = molD.CalcMolFormula(m2,separateIsotopes = False, abbreviateHIsotopes=False)
        
    except:
        chem = np.nan
        m2 = np.nan

    return chem, m2


# Eqns_old = pk.load(open('/nfs/home/6/tempker/GAN/Dataset/pkls/generated_eqs/generated_clean_with_tilda_6-8-20_notrealclean.pkl', 'rb'))
# Eqns = pk.load(open('/nfs/home/6/tempker/aae/generated_text/VAE_generated/5gen_vae_lstm_sample50_exp1_weights.pkl','rb'))

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

EQTable = pd.concat([Rxn, Prd], axis=1)

FQTable = EQTable.stack().value_counts()
FQTable = pd.Series(FQTable.index, name='Smiles_Name').str.replace(' ','').tolist()
Name_List = FQTable
FQTable = [checkbalanceleft(x) for x in FQTable]
FQTable = pd.Series([checkbalanceright(x) for x in FQTable])
Form_List = pd.Series([Smi2Form(x) for x in FQTable])

Form_List = Form_List.tolist()


rep_dict = {Name_List[i]: Form_List[i] for i in range(len(Name_List))}
clean_dict = {v:k for k,v in rep_dict.items()}
clean_dict = {k:v for k,v in clean_dict.items() if pd.notna(k)}


output = open('/nfs/home/6/tempker/aae/generated_text/created_species_lists/both/generated_species_list_for_all_species.pkl','wb')
pk.dump(clean_dict, output)