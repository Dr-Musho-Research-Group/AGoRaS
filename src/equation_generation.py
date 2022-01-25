from keras.preprocessing.sequence import pad_sequences
import numpy as np
from scipy import spatial
from random import randint

def padd_equation(sentence, tokenizer, max_length_of_equation):
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

def sequence_to_smiles(sequence, tokenizer):
    character_list = list(np.vectorize(tokenizer.index_word.get)(sequence))
    characters = [character for character in character_list if character not in ["pad"]]
    return "".join(characters)


def calculate_equations_homology(equation1, equation2, n, encoder, pad_equation = False):
    if pad_equation:
        equation1 = padd_equation(equation1)
        equation2 = padd_equation(equation2)

    encoded_equation1 = encoder.predict(equation1, batch_size = 1)
    encoded_equation2 = encoder.predict(equation2, batch_size = 1)

    return shortest_homology(encoded_equation1, encoded_equation2, n)


def new_equations(homology, generator, latent_dimension, max_length_of_equation, tokenizer):
    for point in homology:
        sequence = generate_equation_sequence(point, latent_dimension, generator)
        indices = reconstruct_indices(sequence, len(tokenizer.word_index), max_length_of_equation)
        print(sequence_to_smiles(indices, tokenizer))
        

def new_equation_generation(homology, generator, latent_dimension, max_length_of_equation, tokenizer):
    list_reactions = []
    for point in homology:
        sequence = generate_equation_sequence(point, latent_dimension, generator)
        indices = reconstruct_indices(sequence, len(tokenizer.word_index), max_length_of_equation)
        equation = sequence_to_smiles(indices, tokenizer)
        list_reactions.append(equation)
    return list_reactions


def generate_equations(dataset, number_of_total_equations, number_of_samples, generator, encoder, latent_dimension, max_length_of_equation, tokenizer)
    equations = []
    while len(equations) < number_of_total_equations:
        equation1 = get_random_equation(dataset)
        equation2 = get_random_equation(dataset)
        homology = calculate_equations_homology(equation1, equation2, number_of_samples, encoder)
        interpolated_equation = new_equation_generation(homology, generator, latent_dimension, max_length_of_equation, tokenizer)
        new_equations = get_unique_equations(interpolated_equation)
        equations.extend(new_equations)
    return list(set(equations))


def get_random_equation(dataset):
        integer = randint(0, len(dataset))
        return dataset[integer].reshape(1,len(dataset[0]))


def get_unique_equations(equations):
    delete_list = ['~~','>>','= ', '[[', ']]', '> >', '[]','()','[ ', '[)', '(]', '~ ~', '\ \ ',
               '###', '..', '==', '# ', '((', '))']
    return [equation for equation in equations if all(i not in equation for i in delete_list)]