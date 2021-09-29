#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:12:39 2020

@author: tempker
"""

import numpy as np
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as molD
import pandas as pd
import pickle as pk

PIK = pk.load(open(r'Original_method_smiles_species_for_rdkit_compare_111920.pkl','rb'))
PIK = PIK.reset_index(drop=True)
species = PIK['cas_number']
smiles = PIK['smiles']


smiles[0] =  '[O]'

mol = [Chem.MolFromSmiles(smile) for smile in smiles]


def calcmolformula(x):
    try:
        x = Chem.MolFromSmiles(x)
        mol = molD.CalcMolFormula(x , separateIsotopes=True, abbreviateHIsotopes=True)
    except:
        mol = np.nan
        
    return(mol)

def addhs(x):
    try:
        mol = Chem.rdmolops.AddHs(x,explicitOnly = True)
    except:
        mol = np.nan
        
    return(mol)

def smilehs(x):
    try:
        mol = Chem.MolToSmiles(x)
    except:
        mol = np.nan
        
    return(mol)

mol_hydrogen = [addhs(x) for x in mol]
mol_hydrogen = [smilehs(x) for x in mol_hydrogen]
species_from_mol = [calcmolformula(x) for x in mol_hydrogen]


compare = pd.concat([PIK ,pd.DataFrame(species_from_mol),pd.DataFrame(mol_hydrogen)], ignore_index=True, axis = 1)
compare.columns = ['Original','Smiles','RDKit_Generated','Smiles_with_Hs']

data = pk.dump(compare, open(r'/nfs/home/6/tempker/GAN/Dataset/pkls/orig_chem_equations_rdkit_generated_chem_equations.pkl', 'wb'))
