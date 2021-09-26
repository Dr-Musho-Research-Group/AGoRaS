import pickle as pk
import pandas as pd
import numpy as np

from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as molD
from rdkit.Chem import AddHs
import os
from glob import glob
os.chdir('/.nfs/home/6/tempker/aae/helper_codes')
from math import gcd
import re

# things need fixed here, mostly absolute paths
PATH = "/.nfs/home/6/tempker/aae/generated_text/VAE_generated"
EXT = "*.pkl"
pkl_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, EXT))]
def openf(pik):
    try:
        with open(pik, 'rb') as f:
                file = pk.load(f)
                return(file)

    except:
        pass
    
    
file = [openf(x) for x in pkl_files]
file = [x for x in file if x]
Eqns = [x for y in file for x in y]
Eqns = list(set(Eqns))
Eqns = [x.strip() for x in Eqns]
delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
               '##', '..', '==', '# ', '((', '))', '++', ' - ','--','-=-']
Eqns = [x for x in Eqns if all(i not in x for i in delete_list)]
Eqns = [x.replace('   ',' + ') for x in Eqns]
Eqns = [x.replace('  ',' ') for x in Eqns]
Eqns = [x.split('>') for x in Eqns]
Eqns = pd.DataFrame(Eqns)

react = Eqns.iloc[:,0]
prod = Eqns.iloc[:,1]

react = react.str.split('~', expand = True)
prod = prod.str.split('~', expand = True)
react = react.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
prod = prod.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

EQTable = pd.concat([react, prod], axis=1)

FQTable = EQTable.stack().value_counts()
FQTable = pd.Series(FQTable.index, name='Smiles_Name').str.replace(' ','').tolist()
Name_List = pd.DataFrame(FQTable)


def AddsomeHs(x):
    try:
        mol = Chem.MolFromSmiles(x)
        molH = AddHs(mol)
        smileH = Chem.MolToSmiles(molH,allHsExplicit = True)
    except:
        smileH = np.nan
        
    return(smileH)


H_name_list = [AddsomeHs(x) for x in FQTable]

dict_hs = dict(zip(FQTable, H_name_list)) 
saveeqs = EQTable

for i in range(len(EQTable.columns)):
    EQTable.iloc[:,i] = EQTable.iloc[:,i].map(dict_hs)

EQTable.insert(1,'plus0','~')
EQTable.insert(3,'plus1','~')
EQTable.insert(5,'plus2','~')
EQTable.insert(7,'plus3','~')
EQTable.insert(9,'plus4','~')
EQTable.insert(11,'equal','>')
EQTable.insert(13,'plus5','~')
EQTable.insert(15,'plus6','~')
EQTable.insert(17,'plus7','~')
EQTable.insert(19,'plus8','~')


import multiprocessing
from io import StringIO




def fn(i):
    return EQTable[i:i+1000].to_csv(index=False, header=False).split('\n')[:-1]

with multiprocessing.Pool() as pool:
    result = []
    for a in pool.map(fn, range(0, len(EQTable), 1000)):
        result.extend(a)
        
        
        
result = [x.replace(',','') for x in result]
result = [x.replace('~~','') for x in result]
result = [x.strip('~') for x in result]
result = [x.replace('~>~','>') for x in result]
result = [x.replace('~>','>') for x in result]
result = [x.replace('>~','>') for x in result]
result = [x.strip('~') for x in result]

remove_list =  ('>')

result = [ele for ele in result if not ele.startswith(remove_list)] 
result = [ele for ele in result if not ele.endswith(remove_list)] 



from balance_function import *

eqns = []
for i in result:
    try:
        b = balance_equation(i)
        eqns.append(b)
    except:
        pass
    
data = pk.dump(eqns, open('/nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_species_transformed_as_of_01132021.pkl', 'wb'))


def listofspecies(Eqns):
    Eqns = [x.replace(' + ', ' ~ ') for x in Eqns]
    
    eqns = pd.Series(Eqns)
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
    FQTable = pd.Series(FQTable.index, name='Smiles_Name').str.replace(' ','')
    Name_List = pd.Series(FQTable)
    FQTable = Name_List.str.lstrip('0123456789')
    FQTable = list(set(FQTable.tolist()))
    return(FQTable)

species_list = listofspecies(eqns)


def splited_lines_generator(lines, n):
        for i in range(0, len(lines), n):
                yield lines[i: i + n]

for index, lines in enumerate(splited_lines_generator(sl, 2)):
         with open('Orig_data_for_musho_to_convert' + str(index) + '.csv', 'w') as f:
             f.write('\n'.join(lines))


import csv

with open('/.nfs/home/6/tempker/Dataset_creation/Dataset/Original_DATA_species_after_balancing_equations_121420.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

data = [''.join(ele) for ele in data]

data = [AddsomeHs(x) for x in data]

formusho = list(set(species_list) - set(data))




with open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/csv_species_list/total_new_species_152021.csv','w') as file:
    file.writelines("%s\n" % x for x in formusho)

def splited_lines_generator(lines, n):
        for i in range(0, len(lines), n):
                yield lines[i: i + n]

for index, lines in enumerate(splited_lines_generator(formusho, 100)):
         with open('/.nfs/home/6/tempker/aae/generated_text/VAE_generated/csv_species_list/total_new_species_152021_' + str(index) + '.csv', 'w') as f:
             f.write('\n'.join(lines))
