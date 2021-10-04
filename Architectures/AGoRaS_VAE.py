"""
Created on Tue Aug  4 10:53:37 2020

@author: Robert Tempke

Architecture for the AGoRaS VAE design for synthetic creation of chemical reaction strings in SMILES format
link to paper here
"""
import pickle as pk
import numpy as np
import os
from keras.optimizers import Adam
from keras.models import Model, Input
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, Activation
import tensorflow as tf
from keras import backend as K
from random import randint
from src.loss_functions import zero_loss, kl_loss
from src.loss_layers import CustomVariationalLayer
from src.utils import sampling, create_model_checkpoint
from src.equation_generation import *
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)

tf.compat.v1.disable_eager_execution()

"""
Please see the `create_tokenized_dataset.py` for instructions on how to prepare the data for machine learning ingestion
"""

path_to_training_data = Path.cwd().joinpath("training_data", "reaction_data.pkl")

with open('reaction_data.pkl', 'rb') as f:
    data = pk.load(f)

dataset = data["training_data"]
tokenizer = data["tokenizer"]


number_of_equations = dataset.shape[0]
np.random.shuffle(dataset)
#current training method is unstable if batch size is not a fraction of the length of training data
training = dataset[:5900].astype(np.int32)
test = dataset[5900:6900].astype(np.int32)



batch_size = 25
epochs = 500
max_length_of_equation = len(dataset[1])
latent_dimension = 350
intermediate_dimension = 500
epsilon_std = 0.1
kl_weight = 0.1
number_of_letters = len(tokenizer.word_index)
learning_rate = 1e-5
optimizer = Adam(lr=learning_rate) 




input = Input(shape=(max_length_of_equation,))
embedded_layer = Embedding(number_of_letters, intermediate_dimension, input_length=max_length_of_equation)(input)
latent_vector = Bidirectional(LSTM(intermediate_dimension, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(embedded_layer)
z_mean = Dense(latent_dimension)(latent_vector)
z_log_var = Dense(latent_dimension)(latent_vector)

z = Lambda(sampling, output_shape=(latent_dimension,))([z_mean, z_log_var])
repeated_context = RepeatVector(max_length_of_equation)
decoder_latent_vector = LSTM(intermediate_dimension, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = Dense(number_of_letters, activation='linear')#softmax is applied in the seq2seqloss by tf #TimeDistributed()
latent_vector_decoded = decoder_latent_vector(repeated_context(z))
input_decoded_mean = decoder_mean(latent_vector_decoded)

    


loss_layer = CustomVariationalLayer()([input, input_decoded_mean])
vae = Model(input, [loss_layer])

vae.compile(optimizer=optimizer, loss=[zero_loss], metrics=[kl_loss])
vae.summary()

#======================= Model training ==============================#
checkpointer = create_model_checkpoint('models', 'vae_balanced_data_run_2')



vae.fit(training, training,
      epochs=epochs,
      batch_size=batch_size,
      validation_data=(test, test), callbacks=[checkpointer])

print(K.eval(vae.optimizer.lr))
K.set_value(vae.optimizer.lr, learning_rate)



vae.save_weights('models/agoras_vae.h5')

# build a model to project inputs on the latent space
encoder = Model(input, z_mean)
encoder.save('models/agoras_encoder.h5')

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dimension,))
_h_decoded = decoder_latent_vector(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)
generator.save('models/agoras_generator.h5')



def generate_indices_from_encoded_space(encoded_vector, generator):
    reconstructed_equation = generator.predict(encoded_vector, batch_size = 1)
    reconstruct_indices = np.apply_along_axis(np.argmax, 1, reconstructed_equation[0])
    return np.max(np.apply_along_axis(np.max, 1, reconstruct_indices[0]))


def test_validation_equation(index_number, data, encoder, generator, tokenizer):
    encoded_equation = encoder.predict(data[index_number:index_number+2,:])
    smiles_indices = generate_indices_from_encoded_space(encoded_equation, generator)
    smiles_equation = list(np.vectorize(tokenizer.index_word.get)(smiles_indices))
    print(f"The reconstructed equation is {''.join(smiles_equation)}")
    original_equation = list(np.vectorize(tokenizer.index_word.get)(data[index_number]))
    print(f"The original equation is {''.join(original_equation)}")


test_validation_equation(200, test, encoder, generator, tokenizer)


#====================== Example ====================================#
equation1=['C=O ~ [OH-] > [CH]=O ~ O']
mysent = padd_equation(equation1, tokenizer, max_length_of_equation)
mysent_encoded = encoder.predict(mysent, batch_size = 1)
print_latent_sentence(mysent_encoded, generator, latent_dimension, max_length_of_equation, number_of_letters, index2word)
print_latent_sentence(find_similar_encoding(mysent_encoded), generator, latent_dimension, max_length_of_equation, number_of_letters, index2word)

equation2=['C.CCCO ~ O=O > CC(=O)C(C)=O ~ [OH-]']
mysent2 = padd_equation(equation2, tokenizer, max_length_of_equation)
mysent_encoded2 = encoder.predict(mysent2, batch_size = 1)
print_latent_sentence(mysent_encoded2, generator, latent_dimension, max_length_of_equation, number_of_letters, index2word)
print_latent_sentence(find_similar_encoding(mysent_encoded2), generator, latent_dimension, max_length_of_equation, number_of_letters, index2word)
print('-----------------')
homology = calculate_equations_homology(equation1, equation2, n, encoder)

new_equation(homology, generator, latent_dimension, max_length_of_equation, tokenizer)
        
generated_equations = []
#A list of common errors to help eliminate bad equations from the generated set. 
delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
               '###', '..', '==', '# ', '((', '))']

while len(generated_equations) < 500000:
    if len(generated_equations) < 500000:
        num1 = randint(0, 6920)
        num2 = randint(0, 6920)
        sent1 = dataset[num1].reshape(1,len(dataset[0]))
        sent2 = dataset[num2].reshape(1,len(dataset[0]))
        newintrp =  new_sents_generation(sent1, sent2, 25, encoder, generator, latent_dimension, max_length_of_equation, number_of_letters, index2word)
        newintrp = [x for x in newintrp if all(i not in x for i in delete_list)]
    gen.extend(newintrp)
    gen = list(set(gen))
    
    print(len(gen))
