import keras
from metrics import F1_score, sensitivity, specificity
import gin


@gin.configurable
def set_adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False):
    ''' set the values for the Adam optimizer'''
    return keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)

@gin.configurable
def set_loss():
    return 'categorical_crossentropy'

@gin.configurable
def set_metrics():
    return [F1_score, sensitivity, specificity, 'accuracy']