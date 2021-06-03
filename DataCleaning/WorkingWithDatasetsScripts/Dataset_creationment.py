import re
from sympy import Matrix, lcm
import pickle as pk
import glob, os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from rdkit import Chem

PIK = '/.nfs/home/6/tempker/GAN/Dataset/pkls/Original_method_smiles_dataframe_12665_eqs_111920.pkl'
with open(PIK, "rb") as f:
        equations = pk.load(f) 
        
        
PIK = r'/nfs/home/6/tempker/GAN/Dataset/pkls/orig_chem_equations_rdkit_generated_chem_equations.pkl'
orig_gen_species = pk.load(open(PIK, 'rb'))



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
    

# tk = Tokenizer(num_words=None,lower=True, filters = "/|>~[\\]()@-+=H",char_level=True)

# # str1 = ''
# # str1 = str1.join(equations)

# tk.fit_on_texts(equations)
# alpha = tk.word_index
# train_sequences = tk.texts_to_sequences(equations)



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

reactantsone = list([reactants[0]])
productone = list([products[0]])



elementList=[]
elementMatrix=[]

for i in range(len(reactantsone)):
    compoundDecipher(reactantsone[i],i,1)
for i in range(len(productone)):
    compoundDecipher(productone[i],i+len(reactantsone),-1)
elementMatrix = Matrix(elementMatrix)
elementMatrix = elementMatrix.transpose()
solution=elementMatrix.nullspace()[0]
