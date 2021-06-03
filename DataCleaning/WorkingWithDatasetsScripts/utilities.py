import re
import string

import numpy as np
import pickle


def save_training_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    print("Saved file training data to %s." % path)


def load_training_data(path):
    with open(path, 'rb') as file:
        saved_data = pickle.load(file)
        file.close()
    print("Loaded file training data from %s." % path)
    return saved_data


def save_model_parameters(path, model, model_type):
    if model_type is "rnn":
        np.savez(path, input_weights=model.input_weights, output_weights=model.output_weights, hidden_weights=model.hidden_weights)
    elif model_type is "lstm":
        np.savez(path, input_weights_g=model.input_weights_g, input_weights_i=model.input_weights_i, input_weights_f=model.input_weights_f, input_weights_o=model.input_weights_o,
                 hidden_weights_g=model.hidden_weights_g, hidden_weights_i=model.hidden_weights_i, hidden_weights_f=model.hidden_weights_f, hidden_weights_o=model.hidden_weights_o,
                 bias_g=model.bias_g, bias_i=model.bias_i, bias_f=model.bias_f, bias_o=model.bias_o,
                 output_weights=model.output_weights, bias_output=model.bias_output)
    print("Saved model parameters to %s." % path)


def load_model_parameters(path, model, model_type):
    npzfile = np.load(path)
    if model_type is "rnn":
        model.input_weights = npzfile["input_weights"]
        model.output_weights = npzfile["output_weights"]
        model.hidden_weights = npzfile["hidden_weights"]
        model.hidden_dimension = model.input_weights.shape[0]
        model.word_dimension = model.input_weights.shape[1]
    elif model_type is "lstm":
        model.input_weights_g, model.input_weights_i, model.input_weights_f, model.input_weights_o = npzfile["input_weights_g"], npzfile["input_weights_i"], npzfile["input_weights_f"], npzfile["input_weights_o"]
        model.hidden_weights_g, model.hidden_weights_i, model.hidden_weights_f, model.hidden_weights_o = npzfile["hidden_weights_g"], npzfile["hidden_weights_i"], npzfile["hidden_weights_f"], npzfile["hidden_weights_o"]
        model.bias_g, model.bias_i, model.bias_f, model.bias_o = npzfile["bias_g"], npzfile["bias_i"], npzfile["bias_f"], npzfile["bias_o"]
        model.output_weights, model.bias_output = npzfile["output_weights"], npzfile["bias_output"]
    print("Loaded model parameters from %s. " % path)
    return model


# Generate a sentence with a Keras model
def keras_generate_sentence(model, max_input_len, word_to_index, index_to_word):

    # Special tokens
    # unknown_token = "UNK"
    sentence_start_token = "$"
    sentence_end_token = "&"

    # We start the sentence with the start token
    sentence_word_ids = np.zeros((1, max_input_len), dtype=int)
    sentence_word_ids[0][0] = word_to_index[sentence_start_token]

    sentence_tokens = []
    word_index = 1

    # Repeat until we get an end token
    for i in range(0, max_input_len):

        # Generate some word predictions
        sentence_probs = model.predict_proba(sentence_word_ids, batch_size=1, verbose=1)
        word_probs = sentence_probs[0][i]

        sampled_word = np.argmax(word_probs)
#=================commented the unknown words section out for now============
        # We don't want to sample unknown words
        # if sampled_word == word_to_index[unknown_token]:
        #     # Remove the unknown token and get second most likely word
        #     word_probs = np.delete(word_probs, sampled_word)
        #     sampled_word = np.argmax(word_probs)
#===========================================================================
        if sampled_word == word_to_index[sentence_end_token]:
            break
        if word_index >= max_input_len:
            break

        sentence_tokens.append(index_to_word[sampled_word])
        sentence_word_ids[0][word_index] = sampled_word
        word_index += 1

    # Convert tokens to string
    sentence = tokens_to_sentence(sentence_tokens)

    # # Write sentences to file
    # with open(file_path, 'a') as file:
    #     file.write(sentence + "\n")

    print("Generated sentence: " + sentence)
    return sentence


# Generate sentence from tokens
def tokens_to_sentence(sentence_tokens):

    sentence = ""

    for i, word in enumerate(sentence_tokens):

        if word is '"' or word in string.punctuation:
            sentence += word
        else:
            sentence += "" + word

    sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentence)
    sentence.lstrip()
    return sentence
