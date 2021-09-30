from keras.preprocessing.sequence import pad_sequences
import numpy as np
from scipy import spatial


def padd_sentence(sentence, tokenizer, max_length_of_equation):
    sequence = tokenizer.texts_to_sequences(sentence)
    return pad_sequences(
        sequence,
        maxlen=max_length_of_equation,
        padding='post',
        value=tokenizer.word_index.get(" "),
    )

# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_vect:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    return sent_vect[maximum]


def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    return [point_one + s * dist_vec for s in sample]


def generate_equation_sequence(vector, latent_dimension, generator):
    equation_vector = np.reshape(vector,[1,latent_dimension])
    return generator.predict(equation_vector)

def reconstruct_indices(sequence, number_of_letters, max_length_of_equation):
    reshaped_sequence = np.reshape(sequence,[max_length_of_equation,number_of_letters])
    return np.apply_along_axis(np.argmax, 1, reshaped_sequence)

def convert_to_smiles(sequence, tokenizer):
    character_list = list(np.vectorize(tokenizer.index_word.get)(sequence))
    characters = [character]


#these two functions do the same thing, this is bad bad programing 
def save_latent_sentence(sent_vect, generator, latent_dimension, max_length_of_equation, tokenizer):
    sent_vect = np.reshape(sent_vect,[1,latent_dimension])
    sent_reconstructed = generator.predict(sent_vect)
    number_of_letters = len(tokenizer.word_index)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_length_of_equation,number_of_letters])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(tokenizer.index_word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['pad']]
    w_list = ''.join(w_list)
    return w_list
    
def print_latent_sentence(sent_vect, generator, latent_dimension, max_length_of_equation, tokenizer):

    sent_vect = np.reshape(sent_vect,[1,latent_dimension])
    sent_reconstructed = generator.predict(sent_vect)
    number_of_letters = len(tokenizer.word_index)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_length_of_equation,number_of_letters])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(tokenizer.index_word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['pad']]
    print(''.join(w_list))
        
def new_sents_interp(sent1, sent2, n, encoder, generator, latent_dimension, max_length_of_equation, tokenizer):
    tok_sent1 = padd_sentence(sent1)
    tok_sent2 = padd_sentence(sent2)
    enc_sent1 = encoder.predict(tok_sent1, batch_size = 1)
    enc_sent2 = encoder.predict(tok_sent2, batch_size = 1)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    number_of_letters = len(tokenizer.word_index)
    for point in test_hom:
        print_latent_sentence(point, generator, latent_dimension, max_length_of_equation, number_of_letters, tokenizer)
        

def new_sents_generation(sent1, sent2, n, encoder, generator, latent_dimension, max_length_of_equation, tokenizer):
    enc_sent1 = encoder.predict(sent1, batch_size = 1)
    enc_sent2 = encoder.predict(sent2, batch_size = 1)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    list_react = []
    for point in test_hom:
        x = save_latent_sentence(point, generator, latent_dimension, max_length_of_equation, tokenizer)
        list_react.append(x)
    return list_react