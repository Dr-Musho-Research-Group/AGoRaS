import os
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5"
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    return ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dimension), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
