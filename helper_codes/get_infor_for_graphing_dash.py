#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:12:05 2021

@author: tempker
"""
import pickle as pk
from rdkit import Chem

PIK = '/nfs/home/6/tempker/Experiment/tsne_3d_results_for_dash_v6.pkl'

with open(PIK, "rb") as f:
    a = pk.load(f)
    
    
smiles = a[2]
mol = [Chem.MolFromSmiles(x) for x in smiles]