#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:53:37 2020

@author: Robert Tempke

Architecture for the AGoRaS VAE design for synthetic creation of chemical reaction strings in SMILES format
link to paper here
"""
import pickle as pk
import numpy as np
import os
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model, Sequential, Input
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
import tensorflow as tf
from keras import backend as K
import pandas as pd
from random import randint
import tensorflow_addons as tfa
from keras.preprocessing.sequence import pad_sequences
from scipy import spatial
from src.loss_functions import zero_loss

os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)

tf.compat.v1.disable_eager_execution()


"""
Please see the DataCleaning directory for instructions on how to prepare the data for machine learning ingestion
"""
#need to replace this with a relative path using pathlib
reaction_data = open('/nfs/home/6/tempker/GAN/Dataset/pkls/Balanced_original_equations_training_sequence_prebalanced_equations__1212020.pkl', 'rb')
reaction_data = pk.load(reaction_data)
dataset = list(reaction_data.values())[0]
wrd2ind = list(reaction_data.values())[1]
ind2wrd = list(reaction_data.values())[2]
vocabulary = list(reaction_data.values())[3]
tokenizer = list(reaction_data.values())[4]

#replace the tokenizer's word index with the one we created
tokenizer.word_index = wrd2ind.copy()



number_of_equations = dataset.shape[0]

np.random.shuffle(dataset)
#current training method is unstable if batch size is not a fraction of the length of training data
training = dataset[:5900].astype(np.int32)
test = dataset[5900:6900].astype(np.int32)



batch_size = 25
max_length_of_equation = len(dataset[1])
latent_dimension = 350
intermediate_dimension = 500
epsilon_std = 0.1
kl_weight = 0.1
number_of_letters = len(vocabulary)
learning_rate = 1e-5
optimizer = Adam(lr=learning_rate) 


def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss

x = Input(shape=(max_length_of_equation,))
x_embed = Embedding(number_of_letters, intermediate_dimension, input_length=max_length_of_equation)(x)
h = Bidirectional(LSTM(intermediate_dimension, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
z_mean = Dense(latent_dimension)(h)
z_log_var = Dense(latent_dimension)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dimension), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dimension,))([z_mean, z_log_var])
repeated_context = RepeatVector(max_length_of_equation)
decoder_h = LSTM(intermediate_dimension, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = Dense(number_of_letters, activation='linear')#softmax is applied in the seq2seqloss by tf #TimeDistributed()
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)





# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_length_of_equation)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tfa.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)#,
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        xent_loss = K.mean(xent_loss)
        kl_loss = K.mean(kl_loss)
        return K.mean(xent_loss + kl_weight * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return K.ones_like(x)
    


loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])

vae.compile(optimizer=optimizer, loss=[zero_loss], metrics=[kl_loss])
vae.summary()

#======================= Model training ==============================#
def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5"
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    return ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

checkpointer = create_model_checkpoint('models', 'vae_balanced_data_run_2')



vae.fit(training, training,
      epochs=500,
      batch_size=batch_size,
      validation_data=(test, test), callbacks=[checkpointer])

print(K.eval(vae.optimizer.lr))
K.set_value(vae.optimizer.lr, learning_rate)



vae.save_weights('models/vae_balanced_data_run_2.h5')
#vae.load_weights('models/vae_seq2seq_test.h5')
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
#encoder.save('models/encoder32dim512hid30kvocab_loss29_val34.h5')

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dimension,))
_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)





index2word = ind2wrd

#test on a validation sentence
sent_idx = 200
sent_encoded = encoder.predict(test[sent_idx:sent_idx+2,:])
x_test_reconstructed = generator.predict(sent_encoded, batch_size = 1)
reconstructed_indexes = np.apply_along_axis(np.argmax, 1, x_test_reconstructed[0])
np.apply_along_axis(np.max, 1, x_test_reconstructed[0])
np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[0]))
word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
print(''.join(word_list))
original_sent = list(np.vectorize(index2word.get)(test[sent_idx]))
print(''.join(original_sent))



# function to parse a sentence
def sent_parse(sentence):
    sequence = tokenizer.texts_to_sequences(sentence)
    padded_sent = pad_sequences(sequence, maxlen=max_length_of_equation, padding='post',value = wrd2ind.get(' ',' '))
    return padded_sent

# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec


def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample


def save_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dimension])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_length_of_equation,number_of_letters])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['pad']]
    w_list = ''.join(w_list)
    return w_list
    
def print_latent_sentence(sent_vect):

    sent_vect = np.reshape(sent_vect,[1,latent_dimension])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_length_of_equation,number_of_letters])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['pad']]
    print(''.join(w_list))
    #print(word_list)       
        
def new_sents_interp(sent1, sent2, n):
    tok_sent1 = sent_parse(sent1)
    tok_sent2 = sent_parse(sent2)
    enc_sent1 = encoder.predict(tok_sent1, batch_size = 1)
    enc_sent2 = encoder.predict(tok_sent2, batch_size = 1)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    for point in test_hom:
        print_latent_sentence(point)
        

def new_sents_generation(sent1, sent2, n):
    enc_sent1 = encoder.predict(sent1, batch_size = 1)
    enc_sent2 = encoder.predict(sent2, batch_size = 1)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    list_react = []
    for point in test_hom:
        x = save_latent_sentence(point)
        list_react.append(x)
    return list_react
        

#====================== Example ====================================#
sentence1=['C=O ~ [OH-] > [CH]=O ~ O']
mysent = sent_parse(sentence1)
mysent_encoded = encoder.predict(mysent, batch_size = 1)
print_latent_sentence(mysent_encoded)
print_latent_sentence(find_similar_encoding(mysent_encoded))

sentence2=['C.CCCO ~ O=O > CC(=O)C(C)=O ~ [OH-]']
mysent2 = sent_parse(sentence2)
mysent_encoded2 = encoder.predict(mysent2, batch_size = 1)
print_latent_sentence(mysent_encoded2)
print_latent_sentence(find_similar_encoding(mysent_encoded2))
print('-----------------')

new_sents_interp(sentence1, sentence2, 5)
        
gen = []
delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
               '###', '..', '==', '# ', '((', '))']

while len(gen) < 500000:
    if len(gen) < 500000:
        num1 = randint(0, 6920)
        num2 = randint(0, 6920)
        sent1 = dataset[num1].reshape(1,len(dataset[0]))
        sent2 = dataset[num2].reshape(1,len(dataset[0]))
        newintrp =  new_sents_generation(sent1, sent2, 25)
        newintrp = [x for x in newintrp if all(i not in x for i in delete_list)]
    gen.extend(newintrp)
    gen = list(set(gen))
    
    print(len(gen))
    
  
org_eqs = pk.load(open('/.nfs/home/6/tempker/GAN/Dataset/pkls/Master_list_wo_startandend_tokens.pkl', "rb") )
    
generated = list(filter(lambda x: x not in org_eqs, gen))

output = open('/nfs/home/6/tempker/aae/generated_text/VAE_generated/vae_balanced_data_run_2_25atatime.pkl','wb')
pk.dump(gen, output)   