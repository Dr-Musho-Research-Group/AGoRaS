
import re
from sympy import Matrix, lcm
import pickle as pk
import glob, os
import pandas as pd
import numpy as np

# os.chdir(r'D:\PhD\Generated Text VAE\P&R from JOULE\eqs_smiles_species\species')


PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_species_transformed_as_of_01132021.pkl'
with open(PIK, "rb") as f:
        equations = pk.load(f)
        
PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_smiles_transformed_as_of_92820.pkl'
with open(PIK, "rb") as f:
        equations_smiles = pk.load(f)


def split_eqs(somelist):
    df = pd.DataFrame(somelist)
    df = df.loc[:,0].str.split(" => ", n = 1, expand = True) 
    df_react = df.iloc[:,0].str.split(" ", n = 8, expand = True) 
    df_prod = df.loc[:,1].str.split(" ", n = 8, expand = True) 
    return df_react, df_prod

def stack_and_count(df1,df2):
    df1 = pd.DataFrame(df1.values.ravel())
    df2 = pd.DataFrame(df2.values.ravel())
    df1 = df1.append(df2)
    counts = df1[0].value_counts()
    return(counts)

def clean_up(df1,df2):
    cols = [2,5]
    df1.drop(df1.columns[cols], axis = 1, inplace = True)
    df1.iloc[:,1] = df2.iloc[:,0]
    df1.iloc[:,3] = df2.iloc[:,1]
    df1.iloc[:,5] = df2.iloc[:,2]
    return df1

def replace_smiles(df):
    df.iloc[:,1] = df.iloc[:,1].map(dict_gibbs)
    df.iloc[:,3] = df.iloc[:,3].map(dict_gibbs)
    df.iloc[:,5] = df.iloc[:,5].map(dict_gibbs)
    df = df.fillna(0)
    df.astype(float)
    
    return df
    
# equations_smiles = [' '.join(ele) for ele in equations_smiles] 
equations_smiles = [x.replace('  ',' ') for x in equations_smiles]
equations_smiles = [x.replace(' = ', ' => ') for x in equations_smiles]


eq_smiles_react, eq_smiles_prod = split_eqs(equations_smiles)
eq_species_react, eq_species_prod = split_eqs(equations)
eqsr = stack_and_count(eq_smiles_react,eq_smiles_prod)

eqsr = eqsr.keys().to_list()
eqsr = [''.join(i) for i in eqsr]


"""

Dumbs everything into diffrent files

"""

# output = open(r'D:\PhD\Generated Text VAE/species_left_after_balance.pkl','wb')
# pk.dump(eqsr, output)

# import csv

# PIK = r'D:\PhD\Generated Text VAE/species_left_after_balance.csv'

# # opening the csv file in 'w+' mode 
# file = open(PIK, 'w+', newline ='') 
  
# # writing the data into the file 
# with file:     
#     write = csv.writer(file) 
#     write.writerows(eqsr)




def openandclean(dict1, dict2, dict3, dict4):
    dict1.pop(0)
    dict2.pop(0)
    dict3.pop(0)
    dict4.pop(0)
    dictt = dict1 + dict2
    dictt = [s.replace('"', '') for s in dictt]
    dictdf = pd.Series(dictt).str.split(":", expand = True) 
    dictname = dictdf.iloc[:,1].str.split(",",expand = True)
    dictname = dictname.iloc[:,0]
    Binding_energy = dictdf.iloc[:,3].str.split(",",expand = True)
    Binding_energy = Binding_energy.iloc[:,0]
    # dictt = {}
    # for x,y in zip(Binding_energy.to_list(),dictname.to_list()):
    #     dictt.setdefault(y,[]).append(x)
    dictt = dict(zip(dictname,Binding_energy))
    # dictt = pd.Series(Binding_energy.values,index = dictname).to_dict()
    return(dictt)

# def Merge(dict1, dict2):
#     res = {**dict1, **dict2}
#     return res

     

dict_gibbs1 = open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/total_new_species_152021_sall_finish.json', 'r')
dict_gibbs2 = open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/species_with_Hs_output_set2.json','r')
dict_gibbs3 = open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/species_with_Hs_output_set3.json','r')
dict_gibbs4 = open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/activation_energy/species_with_Hs_output_set4.json','r')

dict_gibbs = openandclean(dict_gibbs1.readlines(), dict_gibbs2.readlines(),  dict_gibbs3.readlines(),  dict_gibbs4.readlines())


        
eq_species_react, eq_species_prod = clean_up(eq_species_react,eq_smiles_react), clean_up(eq_species_prod,eq_smiles_prod)




eq_gibbs_react, eq_gibbs_prod = replace_smiles(eq_species_react), replace_smiles(eq_species_prod)

#sums the energy

sumofgibb = pd.DataFrame()

sumofgibb['sum_react'] = eq_gibbs_react.iloc[:,0].astype(float) * eq_gibbs_react.iloc[:,1].astype(float) \
    + eq_gibbs_react.iloc[:,2].astype(float) * eq_gibbs_react.iloc[:,3].astype(float)  \
        + eq_gibbs_react.iloc[:,4].astype(float) * eq_gibbs_react.iloc[:,5].astype(float) 
        

sumofgibb['sum_prod'] = eq_gibbs_prod.iloc[:,0].astype(float) * eq_gibbs_prod.iloc[:,1].astype(float) \
    + eq_gibbs_prod.iloc[:,2].astype(float) * eq_gibbs_prod.iloc[:,3].astype(float) \
        + eq_gibbs_prod.iloc[:,4].astype(float) * eq_gibbs_prod.iloc[:,5].astype(float)

sumofgibb['sum_total'] =  sumofgibb['sum_prod'] - sumofgibb['sum_react']
# sumofgibb['sum_total'] = sumofgibb['sum_total']/27.211

sumofgibbgenerated = pd.concat([sumofgibb,pd.DataFrame(equations), pd.DataFrame(equations_smiles)], axis = 1)
sumofgibbgenerated = sumofgibbgenerated[(sumofgibbgenerated != 0).all(1)]

# calculates if there is any species in the original list that are not present
# in the generated list


#wirtes the diffrence from generated and orignal species list to a csv
# import csv
# PIK = '/.nfs/home/6/tempker/aae/generated_text/VAE_generated/original_species_not_included_generated_dataset.csv'

# # opening the csv file in 'w+' mode 
# file = open(PIK, 'w+', newline ='') 
  
# # writing the data into the file 
# with file:     
#     write = csv.writer(file) 
#     write.writerows(gen_dif)

PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_species_balanced.pkl'
with open(PIK, "rb") as f:
        orig_equations = pk.load(f)
        
PIK = '/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_smiles_balanced.pkl'
with open(PIK, "rb") as f:
        orig_equations_smiles = pk.load(f)
        
        
orig_equations_smiles = [x.replace('  ',' ') for x in orig_equations_smiles]
orig_equations_smiles = [x.replace(' = ', ' => ') for x in orig_equations_smiles]


orig_eq_smiles_react, orig_eq_smiles_prod = split_eqs(orig_equations_smiles)
orig_eq_species_react, orig_eq_species_prod = split_eqs(orig_equations)

def cleanup_orig(df1,df2):
    cols = [2]
    df1.drop(df1.columns[cols], axis = 1, inplace = True)
    df1.iloc[:,1] = df2.iloc[:,0]
    df1.iloc[:,3] = df2.iloc[:,1]
    return df1

def replace_smiles_orig(df):
    df.iloc[:,1] = df.iloc[:,1].map(dict_gibbs)
    df.iloc[:,3] = df.iloc[:,3].map(dict_gibbs)
    # df.iloc[:,5] = df.iloc[:,5].map(dict_gibbs)
    df = df.fillna(0)
    df.astype(float)
    
    return df

orig_eq_species_react, orig_eq_species_prod = cleanup_orig(orig_eq_species_react,orig_eq_smiles_react), cleanup_orig(orig_eq_species_prod,orig_eq_smiles_prod)




orig_eq_gibbs_react, orig_eq_gibbs_prod = replace_smiles_orig(orig_eq_species_react), replace_smiles_orig(orig_eq_species_prod)


sumofgibb_orig = pd.DataFrame()

sumofgibb_orig['sum_react'] = orig_eq_gibbs_react.iloc[:,0].astype(float) * orig_eq_gibbs_react.iloc[:,1].astype(float) \
    + orig_eq_gibbs_react.iloc[:,2].astype(float) * orig_eq_gibbs_react.iloc[:,3].astype(float)  
        
        

sumofgibb_orig['sum_prod'] = orig_eq_gibbs_prod.iloc[:,0].astype(float) * orig_eq_gibbs_prod.iloc[:,1].astype(float) \
    + orig_eq_gibbs_prod.iloc[:,2].astype(float) * orig_eq_gibbs_prod.iloc[:,3].astype(float)     

sumofgibb_orig['sum_total'] =  sumofgibb_orig['sum_prod'] - sumofgibb_orig['sum_react']

# sumofgibb_orig['sum_total'] = sumofgibb_orig['sum_total']/27.211

# check = sumofgibb.values.tolist()

# nanwhere = [x for x in range(0,len(check)) if check[x] is 0]
sumofgibb_orig = pd.concat([sumofgibb_orig,pd.DataFrame(orig_equations), pd.DataFrame(orig_equations_smiles)], axis = 1)
sumofgibb_orig = sumofgibb_orig[(sumofgibb_orig != 0).all(1)]


#saves the equations datasets as pkl files
output = open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_species_gibbs_free_energy.pkl','wb')
pk.dump(sumofgibb_orig, output)

output = open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/generated_species_gibbs_free_energy.pkl','wb')
pk.dump(sumofgibbgenerated, output)
