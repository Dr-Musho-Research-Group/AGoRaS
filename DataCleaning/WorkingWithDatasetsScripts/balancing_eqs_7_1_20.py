#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:45:58 2020

@author: tempker
"""

import os
import pickle as pk
import pandas as pd
import numpy as np

os.chdir('/nfs/home/6/tempker')

PIK = '/nfs/home/6/tempker/GAN/Dataset/pkls/reactions_in_chem_equ_format_and_dict.pkl'

with open(PIK, "rb") as f:
        file = pk.load(f)
        
Prd = file[0]
Rxn = file[1]
Rcts = file[2]


count = pd.unique( Rcts.values.ravel() ).astype(str).tolist()

# check = Rcts.head(100)

sort = sorted(count)


zeros = np.zeros(sort.index('10'))
ones = sort[sort.index('10'):sort.index('=>')]
# ones = np.ones(sort.index('Al')-len(zeros))
counter = np.arange(0,len(sort)-sort.index('=>'))

values = np.concatenate((zeros,ones,counter)).astype(float).tolist()
values[-1] = 0.0
res = dict(zip(sort, values)) 


def mapd(df):
    for columns in df:
        df[columns] = df[columns].map(res)


# for columns in Prd:
#     Prd[columns] = Prd[columns].map(res)
