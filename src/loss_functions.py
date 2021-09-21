from keras import backend as K


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss