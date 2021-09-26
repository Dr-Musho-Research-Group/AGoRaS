#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:56:17 2020

@author: tempker
"""


import re
from sympy import Matrix, lcm
import pickle as pk
import glob, os
import pandas as pd
import numpy as np
from rdkit import Chem

PIK = '/.nfs/home/6/tempker/GAN/Dataset/pkls/Original_method_smiles_dataframe_12665_eqs_111920.pkl'
with open(PIK, "rb") as f:
        equations = pk.load(f) 
        
        
PIK = r'/nfs/home/6/tempker/GAN/Dataset/pkls/orig_chem_equations_rdkit_generated_chem_equations.pkl'
orig_gen_species = pk.load(open(PIK, 'rb'))

equations = [x.split('>') for x in equations]


def Extract(lst, i): 
    return [item[i] for item in lst] 

reactants = Extract(equations, 0)
products = Extract(equations, 1)

reactants = [x.split(' ~ ') for x in reactants]
products = [x.split(' ~ ') for x in products]


species = pd.Series(orig_gen_species.Smiles_with_Hs.values,index=orig_gen_species.Smiles).to_dict()



#splits species equations into products and reactants for balancing        
# products = list(file.values())[1].values.tolist()
# reactants = list(file.values())[0].values.tolist()
products = list(map(''.join, products))
reactants = list(map(' '.join, reactants))
products = [x.strip() for x in products]
reactants = [x.strip() for x in reactants]

# reactants = [x.replace('  ', '') for x in reactants]
# products = [x.replace('  ', '') for x in products]

reactants = pd.DataFrame([x.split(' ') for x in reactants]).replace(species).fillna(' ').replace('', ' ')
reactants = reactants.values.tolist()
reactants = list(map(' '.join, reactants))
reactants = [x.strip() for x in reactants]


products = pd.DataFrame([x.split(' ') for x in products]).replace(species).fillna(' ').replace('', ' ')
products = products.values.tolist()
products = list(map(' '.join, products))
products = [x.strip() for x in products]
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
        
c = [x for x in b if str(x) != 'nan']

#load the species list so that the same indices can be dropped together

PIK = '/nfs/home/6/tempker/aae/transformed_data_chem_eq/eqs_with_smile_copies/Orig_smiles_dataset.pkl'
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
output = open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_species_balanced.pkl','wb')
pk.dump(equations, output)

output = open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_smiles_balanced.pkl','wb')
pk.dump(smiles_equations, output)


#saves the equations datasets as text files, makes it easier to read

with open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_species_balanced.txt','w') as file:
    file.writelines("%s\n" % x for x in equations)


with open('/.nfs/home/6/tempker/aae/generated_text/Balanced_equations/orig_smiles_balanced.txt','w') as file:
    file.writelines("%s\n" % x for x in smiles_equations)

