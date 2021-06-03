#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:13:47 2021

@author: tempker
"""

# Balanced_original_equations_with_coefficents_ellim_12102020.pkl



import re
from sympy import Matrix, lcm
import pickle as pk
import glob, os
import pandas as pd
import numpy as np

# os.chdir(r'D:\PhD\Generated Text VAE\P&R from JOULE\eqs_smiles_species\species')


PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_species_transformed_as_of_01132021.pkl'
with open(PIK, "rb") as f:
        equations_gen = pk.load(f)
        
PIK = '/nfs/home/6/tempker/Dataset_creation/Dataset/pkls/Balanced_original_equations_with_coefficents_ellim_01152021.pkl'
with open(PIK, "rb") as f:
        equations_orig = pk.load(f)




def split_eqs(somelist):
    somelist = [x.replace(' > ',' -> ') for x in somelist]
    somelist = [' ' + x for x in somelist]
    #this adds a space behind all the digits and plus signs so i need to do better here
    somelist = [re.sub('( \d+(\.\d+)?)', r'\1 ', x).strip() for x in somelist]
    somelist = [x.replace(' + ',' ~ ') for x in somelist]
    somelist = [re.sub(r' ~ (?!\d)', ' ~ 1 ', x) for x in somelist]
    somelist = [re.sub(r' -> (?!\d)', ' -> 1 ', x) for x in somelist]
    somelist = [x if x[0].isnumeric() else '1 '+x for x in somelist]
    somelist = [x.replace(' ~ ', ' ') for x in somelist]
    df = pd.DataFrame(somelist)
    df = df.loc[:,0].str.split(" -> ", n = 1, expand = True) 
    for i in range(len(df.columns)):
        df.iloc[:,i] = df.iloc[:,i][~df.iloc[:,i].astype(str).str.contains('->')]
    
    df_react = df.iloc[:,0].str.split(" ", n = 25, expand = True) 
    df_prod = df.loc[:,1].str.split(" ", n = 25, expand = True) 
    return df_react, df_prod

def stack_and_count(df1,df2):
    df1 = pd.DataFrame(df1.values.ravel())
    df2 = pd.DataFrame(df2.values.ravel())
    df1 = df1.append(df2)
    counts = df1[0].value_counts()
    counts = counts.keys().to_list()
    counts = [''.join(i) for i in counts]
    counts = list(int_filter(counts))
    
    return(counts)

def clean_up(df1,df2):
    cols = [2,5]
    df1.drop(df1.columns[cols], axis = 1, inplace = True)
    df1.iloc[:,1] = df2.iloc[:,0]
    df1.iloc[:,3] = df2.iloc[:,1]
    df1.iloc[:,5] = df2.iloc[:,2]
    return df1

def int_filter( someList ):
    for v in someList:
        try:
            int(v)
            continue # Skip these
        except ValueError:
            yield v # Keep these

eq_generated_react, eq_generated_prod = split_eqs(equations_gen)
eq_original_react, eq_original_prod = split_eqs(equations_orig)
eqsr_generated = stack_and_count(eq_generated_react,eq_generated_prod)
eqsr_orig = stack_and_count(eq_original_react,eq_original_prod)

eqsr = eqsr_orig + eqsr_generated



dict_gibbs1 = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/total_new_species_2000_sall_finish.json'
dict_gibbs2 = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/total_new_species_152021_sall_finish.json'
dict_gibbs3 = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/need_converted_for_gas_phase_01192021_sall_finish.json'
dict_gibbs4 = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/need_converted_for_gas_phase_01202021_s1_finish.json'
dict_gibbs5 = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/need_converted_for_elements_01222022_s1_finish.json'


def open_json_as_txt(file):
    
    with open(file, 'r') as current:
        lines = current.readlines()
        if not lines:
            print('FILE IS EMPTY')
        else:
            for line in lines:
                b = line
    return lines
            # c.append(b)
lines1 = open_json_as_txt(dict_gibbs1)            
lines2 = open_json_as_txt(dict_gibbs2)
lines3 = open_json_as_txt(dict_gibbs3)
lines4 = open_json_as_txt(dict_gibbs4)
lines5 = open_json_as_txt(dict_gibbs5)

            
species = lines1+lines2+lines3+lines4+lines5
dontdrop = ['Smiles']
species = [x for x in species if any(word in x for word in dontdrop)]
# generated_species = [x for x in lines2 if any(word in x for word in dontdrop)]
# species = orig_species + generated_species

def find_terms(x, start, end):
    start = x.find(start) + len(start)
    end = x.find(r',', start)
    substring = x[start:end]
    return substring
 
Enthalpy =[find_terms( x, r'MSVAMP_Enthalpy":', ',' ) for x in species]
Entropy =[find_terms( x, r'MSVAMP_Entropy":', ',' ) for x in species]
Dipole =[find_terms( x, r'MSVAMP_TotalDipole":', ',' ) for x in species]


def clean_dft(some_list):
    clean = []
    for i in some_list:
        try:
            c = float(i)
            clean.append(c)
        except:
            c = 0.0
            clean.append(c)
    return clean




Enthalpy = clean_dft(Enthalpy)
Entropy = clean_dft(Entropy)
Dipole = clean_dft(Dipole)


smiles =[find_terms( x, r'Smiles":"', r'"' ) for x in species]
smiles = [x.rstrip(r'"') for x in smiles]
    
enthalpy_dict = {smiles[i]: Enthalpy[i] for i in range(len(smiles))} 
entropy_dict = {smiles[i]: Entropy[i] for i in range(len(smiles))} 
dipole_dict = {smiles[i]: Dipole[i] for i in range(len(smiles))} 

"""
run this section if there is a diffrence in what is in the dataset and what is
given
"""
still_need = list(set(eqsr) - set(smiles))

# elements_charges = [x for x in still_need if len(x) < 8]
# elements_charges = [x for x in elements_charges if '->' not in x]


# with open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/csv_species_list/need_converted_for_gas_phase_01222021.csv','w') as file:
#     file.writelines("%s\n" % x for x in still_need)
    
# with open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/csv_species_list/elements_and_elements_plus_charges_need_converted_01222022.csv','w') as file:
#     file.writelines("%s\n" % x for x in elements_charges)
"""
"""


def replace_smiles_with_numbers(df1, df2, some_dict):
    df = pd.concat([df1, df2], axis=1)
    coeff = df.iloc[:,::2]
    species = df.iloc[:,1::2]
    # for i in range(len(species)):
    #     species.iloc[:,i] = species.iloc[:,i].map(some_dict)
    species = species.replace(some_dict) 
    coeff = coeff.T.set_index(np.arange(len(coeff.columns)) * 2, append=True).T
    species = species.T.set_index(np.arange(len(species.columns)) * 2 + 1, append=True).T

    new_df = pd.concat([coeff, species], axis=1).sort_index(1, 1)
    new_df.columns = new_df.columns.droplevel(1)
    for i in range(len(new_df.columns)):
        new_df.iloc[:,i] = new_df.iloc[:,i][~new_df.iloc[:,i].astype(str).str.contains('\[')]
    
    newer_df1 = new_df.iloc[:,0:len(df1.columns)]
    newer_df2 = new_df.iloc[:,len(df1.columns):len(df.columns)]
    
    return newer_df1.astype(dtype=float), newer_df2.astype(dtype=float)

#need to make sure the reactants are first followed by the products
orig_react_enthlapy, orig_prod_enthlapy = replace_smiles_with_numbers(eq_original_react, eq_original_prod, enthalpy_dict)
orig_react_entropy, orig_prod_entropy = replace_smiles_with_numbers(eq_original_react, eq_original_prod, entropy_dict)
orig_react_dipole, orig_prod_dipole = replace_smiles_with_numbers(eq_original_react, eq_original_prod, dipole_dict)

gen_react_enthlapy, gen_prod_enthlapy = replace_smiles_with_numbers(eq_generated_react, eq_generated_prod, enthalpy_dict)
gen_react_entropy, gen_prod_entropy = replace_smiles_with_numbers(eq_generated_react, eq_generated_prod, entropy_dict)
gen_react_dipole, gen_prod_dipole = replace_smiles_with_numbers(eq_generated_react, eq_generated_prod, dipole_dict)


##------------stores converted datasets in case memeory is to much-------------
# PIK = '/nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_DFT_calculated_dataframes_02012021.pkl'

# data = [orig_react_enthlapy, orig_prod_enthlapy, orig_react_entropy, orig_prod_entropy,
#         orig_react_dipole, orig_prod_dipole, gen_react_enthlapy, gen_prod_enthlapy,
#         gen_react_entropy, gen_prod_entropy, gen_react_dipole, gen_prod_dipole]

# with open(PIK, "wb") as f:
#     pk.dump(data, f)

#script for loading the multiple pickles as a list
# with open(PIK, "rb") as f:
#     a = pk.load(f)
##----------------------------------------------------------------------------


def calculate_delta(df1, df2):
    #reactants
    sumofgibb1 = pd.DataFrame()
    df1_coef = df1.iloc[:,::2]
    df1_int = df1.iloc[:,1::2]
    for i in range(len(df1_coef.columns)):
        sumofgibb1.loc[:,i] = df1_coef.iloc[:,i] * df1_int.iloc[:,i]
    sumofgibb1['sum'] = sumofgibb1.sum(axis=1)
    #products
    sumofgibb2 = pd.DataFrame()
    df2_coef = df2.iloc[:,::2]
    df2_int = df2.iloc[:,1::2]
    for i in range(len(df2_coef.columns)):
        sumofgibb2.loc[:,i] = df2_coef.iloc[:,i] * df2_int.iloc[:,i]
    sumofgibb2['sum'] = sumofgibb2.sum(axis=1)
    #products - reactants
    delta = sumofgibb2['sum'] - sumofgibb1['sum']
    
    return delta

orig_delta_enthlapy = calculate_delta(orig_react_enthlapy, orig_prod_enthlapy)
orig_delta_entropy = calculate_delta(orig_react_entropy, orig_prod_entropy)
orig_delta_dipole = calculate_delta(orig_react_dipole, orig_prod_dipole)
orig_delta_gibbs = orig_delta_enthlapy - 293.15 * orig_delta_entropy
orig_delta_gibbs = orig_delta_gibbs/1000

gen_delta_enthlapy = calculate_delta(gen_react_enthlapy, gen_prod_enthlapy)
gen_delta_entropy = calculate_delta(gen_react_entropy, gen_prod_entropy)
gen_delta_dipole = calculate_delta(gen_react_dipole, gen_prod_dipole)
gen_delta_gibbs = gen_delta_enthlapy - 293.15 * gen_delta_entropy
gen_delta_gibbs = gen_delta_gibbs/1000

# PIK = '/nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_deltas_for_graphing_02052021.pkl'

# data = [orig_delta_enthlapy,orig_delta_entropy,orig_delta_dipole,orig_delta_gibbs,
#         gen_delta_enthlapy,gen_delta_entropy,gen_delta_dipole,gen_delta_gibbs]

# with open(PIK, "wb") as f:
#     pk.dump(data, f)


#convert from kcal/mol to eV
d = 4.3363e-2
orig_delta_gibbs  = orig_delta_gibbs*d
gen_delta_gibbs  = gen_delta_gibbs*d


import matplotlib.pyplot as plt
import seaborn as sns


gibbs_gen_mean = gen_delta_gibbs.mean()
gibbs_orig_mean = orig_delta_gibbs.mean()
gibbs_gen_median = gen_delta_gibbs.median()
gibbs_orig_median = orig_delta_gibbs.median()


gensample_gibbs = gen_delta_gibbs.sample(n = 7601)
gensample_dipole = gen_delta_dipole.sample(n = 7601)
gensample_entropy = gen_delta_dipole.sample(n = 7601)
gensample_enthlapy = gen_delta_dipole.sample(n = 7601)

# origsample = orig_equations['sum_total'].sample(n = 1000)

bins = 50
rangehist = (-2, 2)

plt.hist(orig_delta_gibbs,bins = bins, log = False, range = rangehist, alpha = 0.9, label = 'Original Data')
plt.hist(gensample_gibbs ,bins = bins, log = False, range = rangehist, alpha = 0.6,  label = 'Generated Data')
plt.legend(loc='upper left')
plt.xlabel(r"Gibb's Free Energy [eV]")
plt.ylabel('Frequencey')
plt.show()


bins = 20
rangehist = (-100, 100)

plt.hist(orig_delta_entropy,bins = bins, log = False, range = rangehist, alpha = 0.9, label = 'Original Data')
plt.hist(gensample_entropy ,bins = bins, log = False, range = rangehist, alpha = 0.6,  label = 'Generated Data')
plt.legend(loc='upper left')
plt.xlabel(r"Entropy cal/mol")
plt.ylabel('Frequencey')
plt.show()


bins = 20
rangehist = (-25, 25)

plt.hist(orig_delta_enthlapy,bins = bins, log = False, range = rangehist, alpha = 0.9, label = 'Original Data')
plt.hist(gensample_enthlapy ,bins = bins, log = False, range = rangehist, alpha = 0.6,  label = 'Generated Data')
plt.legend(loc='upper left')
plt.xlabel(r"Enthlapy cal/K/mol")
plt.ylabel('Frequencey')
plt.show()


from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer

tk = Tokenizer(num_words=None, char_level=False)
tk.fit_on_texts(eqsr)

# tk2  = Tokenizer(num_words=None, char_level=False)
# tk2.fit_on_texts(diff)

#Do the whole thing, split up after
train_sequences = tk.texts_to_sequences(eqsr)

# Convert string to index
train_sequences = tk.texts_to_sequences(eqsr_orig)
train_sequences2 = tk.texts_to_sequences(eqsr_generated)

df = pd.DataFrame(train_sequences)
df = df.replace(np.nan, 0)

df2 = pd.DataFrame(train_sequences2)
df2 = df2.replace(np.nan, 0)
data_values = df.values

data_values2 = df2.values


tsne = TSNE(n_components=3, verbose=1, perplexity=20, early_exaggeration=50.0,
            learning_rate=800.0, n_iter=1000)
tsne2 = TSNE(n_components=2, verbose=1, perplexity=20, early_exaggeration=50.0,
            learning_rate=500.0, n_iter=1000)
tsne_results = tsne.fit_transform(data_values)
tsne_results2 = tsne2.fit_transform(data_values)

orig_one = tsne_results2[0:2463,0]
orig_two = tsne_results2[0:2463,1]

gen_one = tsne_results2[2463:21595,0]
gen_two = tsne_results2[2463:21595,1]


plt.figure()
plt.scatter(orig_one, orig_two, c = 'b',label = 'Original data')
plt.scatter(gen_one, gen_two, c = 'r',label = 'generated data')
plt.legend()
plt.show()


orig_one = tsne_results[0:2463,0]
orig_two = tsne_results[0:2463,1]
orig_three = tsne_results[0:2463,2]

gen_one = tsne_results[2463:21595,0]
gen_two = tsne_results[2463:21595,1]
gen_three = tsne_results[2463:21595,2]


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = fig.gca(projection = "3d")
ax.scatter(orig_one,orig_two,orig_three, c='b')
ax.scatter(gen_one,gen_two,gen_three, c='r')
plt.show()


#-----------------------------------------------------------------------------
# PIK = '/nfs/home/6/tempker/Experiment/tsne_3d_results_for_dash_v6.pkl'
# data = [tsne_results, tsne_results2, eqsr, orig_species_entropy,gen_species_entropy, orig_species_enthlapy, gen_species_enthlapy]


# with open(PIK, "wb") as f:
#     pk.dump(data, f)
#-----------------------------------------------------------------------------
#export for experiment with ploty
# data = [eqsr_orig, eqsr_generated, enthalpy_dict, enthalpy_dict, data_values, data_values2, eqsr]

# PIK = '/nfs/home/6/tempker/Experiment/ploty_experiment_data.pkl'

# with open(PIK, "wb") as f:
#     pk.dump(data, f)
    
# import plotly.express as px
# from plotly.offline import plot


def clean_data(df):
    some_list = df.iloc[:,0].tolist()
    clean = []
    for i in some_list:
        try:
            c = float(i)
            clean.append(c)
        except:
            c = 0.0
            clean.append(c)
    return pd.DataFrame(clean)



# import plotly.io as pio

# pio.renderers.default='svg'

def replace_species_with_numbers(df,dict1, dict2):
    df_enthlapy = pd.DataFrame(df).replace(dict1)
    df_entropy = pd.DataFrame(df).replace(dict2)
    df_entropy = clean_data(df_entropy)
    df_enthlapy = clean_data(df_enthlapy)
    df_gibbs = df_enthlapy - 293.15 * df_entropy
    df_gibbs = df_gibbs/4.3363e-2
    
    return (df_gibbs.iloc[:,0].tolist())


orig_species_gibbs = replace_species_with_numbers(eqsr_orig, enthalpy_dict, entropy_dict)
gen_species_gibbs = replace_species_with_numbers(eqsr_generated, entropy_dict, enthalpy_dict)
orig_species_entropy = clean_data(pd.DataFrame(eqsr_orig).replace(entropy_dict))
orig_species_enthlapy =  clean_data(pd.DataFrame(eqsr_orig).replace(enthalpy_dict))
gen_species_entropy =  clean_data(pd.DataFrame(eqsr_generated).replace(entropy_dict))
gen_species_enthlapy =  clean_data(pd.DataFrame(eqsr_generated).replace(enthalpy_dict))




# fig = px.scatter(tsne_results, x= 0, y =1)
# fig.show()
# size_orig = [np.abs(x)/10000 for x in orig_species_gibbs]
# size_gen = [np.abs(x)/10000 for x in gen_species_gibbs]

# size_orig = [(float(i)-min(orig_species_gibbs))/(max(orig_species_gibbs)-min(orig_species_gibbs)) * 10 for i in orig_species_gibbs]
# size_gen = [(float(i)-min(orig_species_gibbs))/(max(orig_species_gibbs)-min(orig_species_gibbs)) * 10 for i in gen_species_gibbs]

# size_orig = [(np.abs(x)**2)/1000000000 for x in orig_species_gibbs]
# size_gen = [(np.abs(x)**2)/100000000 for x in gen_species_gibbs]
size_orig = [(np.abs(x))/1000 for x in orig_species_gibbs]
size_gen = [(np.abs(x))/1000 for x in gen_species_gibbs]

# x = list(orig_one)
# y = list(orig_two)
fig = plt.figure()
plt.scatter(list(orig_one),list(orig_two) , s = size_orig, c='b',label = 'original data', alpha = 1.0)

# print(s)
plt.scatter(list(gen_one),list(gen_two), s = size_gen, c='r',label = 'generated data', alpha = 0.7)
plt.legend()
# plt.annotate('7,601 Original Reactions', xy = (-42, 15), xytext = (-59,50),fontsize = 22,
#              arrowprops=dict(facecolor='black', shrink=0.1))
# plt.annotate('Generated Reactions', xy = (-20, -20), xytext = (-59,-55),fontsize = 22,
#               arrowprops=dict(facecolor='black', shrink=0.1))
plt.grid()
# plt.savefig('t-SNE_orig_data')
plt.show()

# x = [0,2,4,6,8,10]
# y = [0]*len(x)
# s = [20*4**n for n in range(len(x))]
# plt.scatter(x,y,s=s)
# plt.show()