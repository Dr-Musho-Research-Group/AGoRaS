#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:27:12 2020

@author: tempker
"""

import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np
import pandas as pd

PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_species_gibbs_free_energy.pkl'
with open(PIK, "rb") as f:
        orig_equations = pk.load(f)
        
PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/generated_species_gibbs_free_energy.pkl'
with open(PIK, "rb") as f:
        gen_equations = pk.load(f)
        
# gen = gen_equations[gen_equations['sum_total'] > -3e+07]
        
# combo =[orig_equations['sum_total'], gen['sum_total']]
# headers  = ['Original equations', 'Generated Equations']    
# sumtotal = pd.concat(combo, axis =1, keys = headers)


# sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# compare =sns.violinplot(data = sumtotal/27.211)
# compare.set(ylabel = 'Gibbs Free Energy of Equations [Ha]', xlabel = 'Density')


gen_figure = sns.violinplot( y =gen_equations['sum_total'])
gen_figure.set(ylabel = 'Gibbs Free Energy of Generated Equations [Ha]', xlabel = 'Density')

orig_figure= sns.violinplot( y=orig_equations['sum_total'])
orig_figure.set(ylabel = 'Gibbs Free Energy of Original Equations [Ha]', xlabel = 'Density')






gibbs_gen_mean = gen_equations['sum_total'].mean()
gibbs_orig_mean = orig_equations['sum_total'].mean()
gibbs_gen_median = gen_equations['sum_total'].median()
gibbs_orig_median = orig_equations['sum_total'].median()


gensample = gen_equations['sum_total'].sample(n = 2935)
# origsample = orig_equations['sum_total'].sample(n = 1000)

bins = 50
rangehist = (-100, 100)

# plt.hist(origsample,bins = bins, log = False, range = rangehist, alpha = 0.9, label = 'Original Data')
# plt.hist(gensample ,bins = bins, log = False, range = rangehist, alpha = 0.9,  label = 'Generated Data')


# plt.hist(orig_equations['sum_total'],bins = bins, range = (-8e6, 0), log = True, density = False, alpha = 0.9, label = 'Original Data')
plt.hist(gen_equations['sum_total'] ,bins = bins, range = rangehist, log = True, density = False, alpha = 0.9,  label = 'Generated Data')

plt.hist(orig_equations['sum_total'],bins = bins, log = False, range = rangehist, alpha = 0.9, label = 'Original Data')
plt.hist(gensample ,bins = bins, log = False, range = rangehist, alpha = 0.9,  label = 'Generated Data')

plt.legend(loc='upper left')
plt.show()