#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:15:45 2020

@author: tempker
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:10:23 2020

@author: rstem
"""

import pandas as pd
import numpy as np
import pickle as pk
import os
os.chdir(r'/.nfs/home/6/tempker/GAN/Dataset')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utilities import *


data_path = r'/nfs/home/6/tempker/GAN/Dataset/pkls/reactions-training-data-lstm.pkl'


df_lower = pk.load( open( "/nfs/home/6/tempker/GAN/Dataset/pkls/Older_df_that_did_not_work/df_with_all_lowercase.pkl", "rb" ) )

# sentence_start_token = "$"
# sentence_end_token = "&"
# sentences = ["%s %s %s" % (sentence_start_token, df_lower, sentence_end_token) for df_lower in df_lower]
sentences = df_lower
   

        
# =======================Convert string to index================
tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tk.fit_on_texts(sentences)


# =======================Create a new vocabulary================

str1 = ''
str1 = str1.join(sentences)



res = {i : str1.count(i) for i in set(str1)}
alphabet = list(res.keys())[:]
str2 = ''
alphabet = str2.join(alphabet)

char_dict = {}
for i, char in enumerate(alphabet):
    char_dict[char] = i + 1

# Use char_dict to replace the tk.word_index
tk.word_index = char_dict.copy()
# Add 'UNK' to the vocabulary
tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
rev_subs = { v:k for k,v in tk.word_index.items()}



# Convert string to index
train_sequences = tk.texts_to_sequences(sentences)



checking = pd.DataFrame(train_sequences)
df_master = checking.replace(rev_subs)
df_master = df_master.replace(np.nan,'')
tk_sentences = df_master.values.tolist()


# saved = pk.dump(tk_sentences, open("padded_sentences_already_split.pkl",'wb'))

tk_sentencess = []
for i in range(len(tk_sentences)):
    tk = [x for x in tk_sentences[i] if str(x) != '']
    tk_sentencess.append(tk)


# # Padding
# train_data = pad_sequences(train_sequences, maxlen=100, padding='pre')
# train_data = np.array(train_data, dtype='float32')


# Example and labeled training sets
x_train = []
y_train = []
# Create the training data
for sentence in tk_sentencess:
    x = []
    y = []
    # All but the SENTENCE_END token
    for word in sentence[: -1]:
        x.append(char_dict[word])
    # All but the SENTENCE_START token
    for word in sentence[1:]:
        y.append(char_dict[word])

    x_train.append(x)
    y_train.append(y)


x_padded = pad_sequences(x_train, maxlen=None, padding='post', value=0)
y_padded = pad_sequences(y_train, maxlen=None, padding='post', value=0)


num_sentences = x_padded.shape[0]
print("Number of Sentences: ", num_sentences)
max_input_len = x_padded.shape[1]
print("Max Sentence length: ", max_input_len)

vocabulary = [(k, v) for k, v in char_dict.items()] 

# Save data to file
data = dict(
    x_train=x_train,
    y_train=y_train,
    x_padded=x_padded,
    y_padded=y_padded,
    word_to_index=char_dict,
    index_to_word=rev_subs,
    vocabulary=vocabulary,
    num_sentences=num_sentences,
    max_input_len=max_input_len)

print("Saving training data")
try:
    save_training_data(data_path, data)
except FileNotFoundError as err:
    print("Error saving data " + str(err))