#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:53:37 2020

@author: tempker
"""
import pickle as pk
import numpy as np
import os
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint

os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)
from keras.optimizers import Adam
from keras.models import Model, Sequential, Input
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
# from keras.layers import Concatenate, Dropout, BatchNormalization
import tensorflow as tf
from keras import backend as K
import pandas as pd
from random import randint
import tensorflow_addons as tfa
from keras.preprocessing.sequence import pad_sequences
from scipy import spatial

# physical_devices = tf.config.list_physical_devices('GPU')
tf.compat.v1.disable_eager_execution()
# tf.config.experimental.set_memory_growth(physical_devices, True)
# tf.compat.v1.estimator.tpu.TPUEstimator(use_tpu=True, eval_batch_size= 100)

# BATCH_SIZE = 10
# TRAINING_RATIO = 5
# GRADIENT_PENALTY_WEIGHT = 1


# Load data
# with open('/nfs/home/6/tempker/GAN/Dataset/pkls/reactions-training-data.pkl', 'rb') as f: 
#     train_data, wrd2ind, ind2wrd, vocabulary = pk.load(f)
data_GAN = open('/nfs/home/6/tempker/GAN/Dataset/pkls/Balanced_original_equations_training_sequence_prebalanced_equations__1212020.pkl', 'rb')

data_GAN = pk.load(data_GAN)
train_data = list(data_GAN.values())[0]
wrd2ind = list(data_GAN.values())[1]
ind2wrd = list(data_GAN.values())[2]
vocabulary = list(data_GAN.values())[3]
tokenizer = list(data_GAN.values())[4]

tokenizer.word_index = wrd2ind.copy()


np.random.shuffle(train_data)

num_eqs = train_data.shape[0]

# test_split = 0.2

# num_test_samples = math.ceil(num_eqs * test_split)

# test = train_data[:num_test_samples].astype(np.int32)
# training = train_data[num_test_samples:].astype(np.int32)

training = train_data[:5900].astype(np.int32)
test = train_data[5900:6900].astype(np.int32)

batch_size = 20

max_len = len(train_data[1])
# emb_dim = 250
latent_dim = 250
intermediate_dim = 500
epsilon_std = 0.01
kl_weight = 0.01
num_sampled=25
act = ELU()
nb_letters = len(vocabulary)
learnr = 1e-5


x = Input(shape=(max_len,))
x_embed = Embedding(nb_letters, intermediate_dim, input_length=max_len)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
#h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
#h = Dropout(0.2)(h)
#h = (intermediate_dim, activation='linear')(h)
#h = act(h)
#h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# we instantiate these layers separately so as to reuse them later
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = Dense(nb_letters, activation='linear')#softmax is applied in the seq2seqloss by tf #TimeDistributed()
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tfa.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)#,
                                                     #softmax_loss_function=softmax_loss_f), axis=-1)#,
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
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)
    
def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
opt = Adam(lr=learnr) 
vae.compile(optimizer='adam', loss=[zero_loss], metrics=[kl_loss])
vae.summary()

#======================= Model training ==============================#
def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5" 
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    return checkpointer

checkpointer = create_model_checkpoint('models', 'new_training_data_trial_2')



vae.fit(training, training,
      epochs=500,
      batch_size=batch_size,
      validation_data=(test, test), callbacks=[checkpointer])



# vae.fit(training, training,
#      epochs=100,
#      batch_size=batch_size)


print(K.eval(vae.optimizer.lr))
K.set_value(vae.optimizer.lr, learnr)



# vae.save('models/vae_lstm_sample50_exp1.h5')
vae.save_weights('models/new_training_data_trial_2.h5')
#vae.load_weights('models/vae_seq2seq_test.h5')
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
#encoder.save('models/encoder32dim512hid30kvocab_loss29_val34.h5')

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
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


"""
FIX THIS WITH IMPORTING THE TK FROM BEFORE
"""

#=================== Sentence processing and interpolation ======================#
# function to parse a sentence
# def sent_parse(sentence, mat_shape):
#     sequence = tokenizer.texts_to_sequences(sentence)
#     # pad_sequences(train_sequences, maxlen=None, padding='post', value = 49)
#     padded_sent = pad_sequences(sequence, maxlen=max_len, padding='post', value = 49)
#     return padded_sent#[padded_sent, sent_one_hot]

# function to parse a sentence
def sent_parse(sentence):
    sequence = tokenizer.texts_to_sequences(sentence)
    padded_sent = pad_sequences(sequence, maxlen=max_len, padding='post',value = wrd2ind.get(' ',' '))
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


# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample



# input: original dimension sentence vector
# output: sentence text
def save_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dim])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_len,nb_letters])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['pad']]
    w_list = ''.join(w_list)
    return w_list
    
def print_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dim])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_len,nb_letters])
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
sentence1=['[CH3][O][NH2] ~ [Cl] > [NH2][CH2][O] ~ [ClH]']
mysent = sent_parse(sentence1)
mysent_encoded = encoder.predict(mysent, batch_size = 1)
print_latent_sentence(mysent_encoded)
print_latent_sentence(find_similar_encoding(mysent_encoded))

sentence2=['[CH3][C]([CH3])([CH3])[OH] ~ [O-][OH] > [CH3][C]([CH3])([CH3])[O] ~ [OH][OH]']
mysent2 = sent_parse(sentence2)
mysent_encoded2 = encoder.predict(mysent2, batch_size = 1)
print_latent_sentence(mysent_encoded2)
print_latent_sentence(find_similar_encoding(mysent_encoded2))
print('-----------------')

new_sents_interp(sentence1, sentence2, 5)
        
# check = [index2word[x] for x in mysent.tolist()]




# gen = []

# while len(gen) < 500000:
#     if len(gen) < 500000:
#         num1 = randint(0, 32057)
#         num2 = randint(0, 32057)
#         sent1 = train_data[num1].reshape(1,len(train_data[0]))
#         sent2 = train_data[num2].reshape(1,len(train_data[0]))
#         newintrp =  new_sents_generation(sent1, sent2, 10)
#     gen.extend(newintrp)
#     # gen = list(map(''.join,gen))
#     # gen = [''.join(x) for x in gen]
    
    

gen = []
delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
               '###', '..', '==', '# ', '((', '))']

while len(gen) < 1000000:
    if len(gen) < 1000000:
        num1 = randint(0, 6920)
        num2 = randint(0, 6920)
        sent1 = train_data[num1].reshape(1,len(train_data[0]))
        sent2 = train_data[num2].reshape(1,len(train_data[0]))
        newintrp =  new_sents_generation(sent1, sent2, 5)
        newintrp = [x for x in newintrp if all(i not in x for i in delete_list)]
    gen.extend(newintrp)
    gen = list(set(gen))
    
    print(len(gen))
    
  
org_eqs = pk.load(open('pkls/Balanced_original_equations_without_coefficents__1212020.pkl', "rb") )
    
generated = list(filter(lambda x: x not in org_eqs, gen))

output = open('/nfs/home/6/tempker/aae/generated_text/VAE_generated/new_training_data_trial_1.pkl','wb')
pk.dump(gen, output)   
