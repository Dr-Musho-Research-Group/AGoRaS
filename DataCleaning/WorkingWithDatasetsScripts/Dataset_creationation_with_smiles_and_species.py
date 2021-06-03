#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:38:36 2020

@author: tempker
"""

import re
from sympy import Matrix, lcm
import pickle as pk
import glob, os
import pandas as pd
import numpy as np
from rdkit import Chem

"""
Use the following if you only want to balance a single equation
"""
PIK = '/nfs/home/6/tempker/aae/transformed_data_chem_eq/eqs_with_smile_copies/eqs_species_alleqs_asof_9820.pkl'
with open(PIK, "rb") as f:
        file = pk.load(f)
        
        
#splits species equations into products and reactants for balancing        
products = list(file.values())[1].values.tolist()
reactants = list(file.values())[0].values.tolist()
products = list(map(' '.join, products))
reactants = list(map(' '.join, reactants))


# -----------------------------------------------------------------------------

#functions for cleaning everything up
    
def merge_list(somelist):    
    
    value = [[' '.join(x) for x in y] for y in somelist]
    cleanlist = [x for y in value for x in y]
    cleanlist = cleanup(cleanlist)
   
    return(cleanlist)

def cleanup(listofeqs):
    empty = []
    for i in range(len(listofeqs)):
        cle=listofeqs[i].strip().split(" ")
        empty.append(cle)
        
    return(empty)

#=======for multiple files=========
# products = merge_list(p)
# reactants = merge_list(r)
#=================================


products = cleanup(products)
reactants = cleanup(reactants)

#have to drop the "-" or the balancing code gets messed up

products = [[x.replace('-','') for x in y] for y in products]
reactants = [[x.replace('-','') for x in y] for y in reactants]




def addToMatrix(element, index, count, side):
    if(index == len(elementMatrix)):
       elementMatrix.append([])
       for x in elementList:
            elementMatrix[index].append(0)
    if(element not in elementList):
        elementList.append(element)
        for i in range(len(elementMatrix)):
            elementMatrix[i].append(0)
    column=elementList.index(element)
    elementMatrix[index][column]+=count*side
    
def findElements(segment,index, multiplier, side):
    elementsAndNumbers=re.split('([A-Z][a-z]?)',segment)
    i=0
    while(i<len(elementsAndNumbers)-1):#last element always blank
          i+=1
          if(len(elementsAndNumbers[i])>0):
            if(elementsAndNumbers[i+1].isdigit()):
                count=int(elementsAndNumbers[i+1])*multiplier
                addToMatrix(elementsAndNumbers[i], index, count, side)
                i+=1
            else:
                addToMatrix(elementsAndNumbers[i], index, multiplier, side)        
    
def compoundDecipher(compound, index, side):
    segments=re.split('(\([A-Za-z0-9]*\)[0-9]*)',compound)    
    for segment in segments:
        if segment.startswith("("):
            segment=re.split('\)([0-9]*)',segment)
            multiplier=int(segment[1])
            segment=segment[0][1:]
        else:
            multiplier=1
        findElements(segment, index, multiplier, side)


b = []
c = []
    
for x,y in zip(products,reactants):

    try:

        elementList=[]
        elementMatrix=[]

            
        for i in range(len(y)):
            compoundDecipher(y[i],i,1)
        for i in range(len(x)):
            compoundDecipher(x[i],i+len(y),-1)
        elementMatrix = Matrix(elementMatrix)
        elementMatrix = elementMatrix.transpose()
        solution=elementMatrix.nullspace()[0]
        multiple = lcm([val.q for val in solution])
        solution = multiple*solution
        coEffi=solution.tolist()
        output=""
        for i in range(len(y)):
            output+=str(coEffi[i][0])+" " +y[i]
            if i<len(y)-1:
                output+=" + "
        output+=" => "
        for i in range(len(x)):
            output+=str(coEffi[i+len(y)][0])+" "+x[i]
            if i<len(x)-1:
                output+=" + "
        b.append(output)

    except:
        output = np.nan
        b.append(output)

#load the species list so that the same indices can be dropped together

PIK = '/nfs/home/6/tempker/aae/transformed_data_chem_eq/eqs_with_smile_copies/eqs_smiles_alleqs_asof_9820.pkl'
with open(PIK, "rb") as f:
        file = pk.load(f)
        
products = list(file.values())[1].values.tolist()
reactants = list(file.values())[0].values.tolist()
products = list(map(' '.join, products))
reactants = list(map(' '.join, reactants))

products = cleanup(products)
reactants = cleanup(reactants)
      
#creates a dataset that is identical to the species one just in smiles form
#we will need this to subout the enthlapy
Species_list = [a + list('=') + b for a, b in zip(reactants,products)]

#find all indices where there is a nan value that needs droped 
nanwhere = [x for x in range(0,len(b)) if b[x] is np.nan]



"""
not fast enought try something else
"""

#drops all species indices that have nan's
c = [i for n, i in enumerate(b) if n not in nanwhere]

#drops all smiles that were droped out of the species list
Species_listt =  [i for n, i in enumerate(Species_list) if n not in nanwhere]




#create a list of indices from the species dataset that is not valid so we can
#drop them from both datasets
d = []
cc = d.append([x for x in range(0,len(c)) if '-' in c[x]])
cc = d.append([x for x in range(0,len(c)) if '0' in c[x]])
d = [j for i in d for j in i]

#drops the indices that are not valid
equations = [i for n, i in enumerate(c) if n not in d]
smiles_equations = [i for n, i in enumerate(Species_listt) if n not in d]
smiles_equations = [' '.join(ele) for ele in smiles_equations] 



#saves the equations datasets as pkl files
output = open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_species_transformed_as_of_92820.pkl','wb')
pk.dump(equations, output)

output = open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_smiles_transformed_as_of_92820.pkl','wb')
pk.dump(smiles_equations, output)


#saves the equations datasets as text files, makes it easier to read

with open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_species_transformed_as_of_92820.txt','w') as file:
    file.writelines("%s\n" % x for x in equations)


with open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/all_eqs_smiles_transformed_as_of_92820.txt','w') as file:
    file.writelines("%s\n" % x for x in smiles_equations)



#================ This section splits the smiles dataset up so that we can
#           find the number of unique species in the datast =================


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
    counts = df1.iloc[:,0].value_counts()
    return(counts)


equations_smiles = [''.join(ele) for ele in smiles_equations] 
equations_smiles = [x.replace('  ',' ') for x in equations_smiles]
equations_smiles = [x.replace(' = ', ' => ') for x in equations_smiles]

#splits dataset into reaction and products so they can be stacked and counted
eq_smiles_react, eq_smiles_prod = split_eqs(equations_smiles)

#creates a list of unique smiles formated species
eqsr = stack_and_count(eq_smiles_react,eq_smiles_prod)
eqsr = eqsr.keys().to_list()


output = open('/.nfs/home/6/tempker/aae/generated_text/species_left_after_balance_with_Hs.pkl','wb')
pk.dump(eqsr, output)

import csv

PIK = '/.nfs/home/6/tempker/aae/generated_text/species_left_after_balance_with_Hs.csv'

def write_to_csv(list_of_eqs):
    with open(PIK, 'w') as csvfile:
        for domain in list_of_eqs:
            csvfile.write(domain + '\n')

write_to_csv(eqsr)

#have to create one without Hs too because why not

def removeHs(somelist):    

    mols = [Chem.MolFromSmiles(smile) for smile in somelist]    
    mol = [Chem.RemoveHs(x) for x in mols]
    m = [Chem.MolToSmiles(x) for x in mol]
    
    return m

eqsr_noHs = removeHs(eqsr)

output = open('/.nfs/home/6/tempker/aae/generated_text/species_left_after_balance_without_Hs.pkl','wb')
pk.dump(eqsr_noHs, output)


PIK = '/.nfs/home/6/tempker/aae/generated_text/species_left_after_balance_without_Hs.csv'


write_to_csv(eqsr_noHs)


