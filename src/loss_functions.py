from keras import backend as K


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)
