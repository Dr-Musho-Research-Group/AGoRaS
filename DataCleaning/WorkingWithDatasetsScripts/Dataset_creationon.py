
import pandas as pd
import numpy as np
import pickle as pk
import os
import pubchempy as pcp
os.chdir(r'/.nfs/home/6/tempker/GAN/Dataset/csv')


species = pd.read_csv(r'species.csv')

# cas_numbers = species['cas_number'].values.tolist()
iupac_name = species['iupac_name'].values.tolist()
preferred_name = species['preferred_name'].values.tolist()
common_name = species['common_name'].values.tolist()
# aliases = species['aliases'].values.tolist()
# chem_formula = species['chemical_formula'].values.tolist()

def cas_fixer(cas_num):
    cas_num = abs(cas_num)
    cas_str = str(cas_num)
    lds = len(cas_str)

    ans = cas_str[:lds-3] + '-' + cas_str[lds-3:lds-1] + '-' + cas_str[lds-1]
    return ans
    

def aliases_fixer(df):
    

    aliases = df.str.replace('mixture of E- and Z-isomers', '')
    aliases = aliases.str.replace('(mixture of E-and Z-isomers)', '')
    aliases = aliases.str.replace('\(\)', ';')
    aliases = aliases.str.split(';', 6, expand = True)
    aliases_rows = pd.DataFrame()
    i = 0
    for i in range(len(aliases.columns)):
        aliases_row1 = aliases.iloc[:,i].str.split(', ',expand = True)
        aliases_rows = pd.concat([aliases_rows, aliases_row1], axis = 1)
  
    i = 0
    for i in range(len(aliases_rows.columns)):
        aliases_rows.iloc[:,i] = aliases_rows.iloc[:,i].astype(str) + ' '
        aliases_rows.iloc[:,i] = aliases_rows.iloc[:,i].str.replace('- ', '')
        aliases_rows.iloc[:,i] = aliases_rows.iloc[:,i].str.replace(', ', '')
        aliases_rows.iloc[:,i] = aliases_rows.iloc[:,i].str.replace(',  ', '')
        
        aliases_rows.iloc[:,i] = aliases_rows.iloc[:,i].str.strip(' ')
        
    # aliases_rows = aliases_rows.values.tolist()    
    return aliases_rows
   

aliases = aliases_fixer(species['aliases'])   

cas_numbers = [cas_fixer(x) for x in species['cas_number'].values.tolist()]


# def cas_converter(somelist):
#     converted = []
#     for x in somelist:
#         try: 
#             converting = pcp.get_compounds(x, 'name')
#             conv_smiles = converting[0].canonical_smiles
#             converted.append(conv_smiles)
#         except:
#             converted.append(np.nan)
        
#     return converted

 
def name_converter(somelist):
    converted = []
    for x in somelist:
        try: 
            converting = pcp.get_compounds(x, 'name')
            conv_smiles = converting[0].isomeric_smiles
            converted.append(conv_smiles)
        except:
            converted.append(np.nan)
        
    return converted

def chem_converter(somelist):
    converted = []
    for x in somelist:
        try: 
            converting = pcp.get_compounds(x, 'formula')
            conv_smiles = [i.isomeric_smiles for i in converting]
            converted.append(conv_smiles)
        except:
            converted.append(np.nan)
        
    return converted
       

cas_converted = name_converter(cas_numbers)
iupac_converted = name_converter(iupac_name)
preferred_converted = name_converter(preferred_name)
common_converted = name_converter(common_name)

converted = pd.DataFrame({'iupac_converted':iupac_converted,
                          'preferred_converted':preferred_converted,
                          'common_converted':common_converted,
                          'cas_converted':cas_converted})   
 
output = open(r'/.nfs/home/6/tempker/GAN/Dataset/pupchem_conversions_11_18_20.pkl','wb')
pk.dump(converted, output)


aliases_converted_columns = pd.DataFrame()
i=0
for i in range(len(aliases.columns)):
# for i in range(1):
    
    aliases_column = aliases.iloc[:,i].values.tolist()
    aliases_converted = name_converter(aliases_column)
    aliases_converted = pd.DataFrame(aliases_converted)
    aliases_converted_columns = pd.concat([aliases_converted_columns, aliases_converted], axis = 1)
       



aliases_converted_columns = []
i=0
for i in range(len(aliases.columns)):
   
    aliases_column = aliases.iloc[:,i].values.tolist()
    for x in aliases_column:
        aliases_converted = name_converter(x)
        # aliases_converted = pd.DataFrame(aliases_converted)
        aliases_converted_columns.append(aliases_converted)
        # print(x, aliases_converted)
        # aliases_converted_columns = pd.concat([aliases_converted_columns, aliases_converted], axis = 1)
       


output = open(r'/.nfs/home/6/tempker/GAN/Dataset/alieases_pupchem_conversions_11_18_20.pkl','wb')
pk.dump(aliases_converted_columns, output)



# aliases_converted = name_converter(aliases)
# chem_form_conv = chem_converter(chem_formula)

# converted = pd.DataFrame({'iupac_converted':iupac_converted,
#                           'preferred_converted':preferred_converted,
#                           'common_converted':common_converted,
#                           'aliases_converted':aliases_converted})

# concheck = converted.bfill(axis=1).ffill(axis=1)
# concheck['iupac_converted'].isna().sum()

# sub_dict = {species['cas_number'].values.tolist()[i]:concheck['iupac_converted'].values.tolist()[i]
#             for i in range(len(species['cas_number'].values.tolist()))}


# output = open(r'D:\PhD\Dataset\FROM_NIST\pubchem_dict_conversion_of_species_to_smiles_93020.pkl','wb')
# pk.dump(sub_dict, output)