import keras
from metrics import F1_score, sensitivity, specificity, muilticlass_logloss
import gin


@gin.configurable
def set_adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False):
    ''' set the values for the Adam optimizer'''
    return keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)

@gin.configurable
def set_loss():
    return 'categorical_crossentropy'

@gin.configurable
def set_metrics():
    return [F1_score, sensitivity, specificity, muilticlass_logloss]

@gin.configurable
def comp_model(model=None, **kwargs):
    k_opt = set_adam(**kwargs)
    k_loss = set_loss()
    k_metrics = set_metrics()
    model.compile(optimizer=k_opt, loss=k_loss, metrics=k_metrics)
