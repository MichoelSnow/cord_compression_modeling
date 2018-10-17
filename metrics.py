from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
import numpy as np
import sklearn
import gin

class ConfusionMatrix(Callback):
    def __init__(self, val_gen=None, batch_size=None):
        # !!!! A bug in keras keeps self.validation (in the Callback class) from ever being set with a generator,
        # used classes from the validation generator instead
        super().__init__()
        val_gen.reset()
        self.validation_data = val_gen
        self.validation_labels = val_gen.classes
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        print("Calculating confusion matrix")
        predicted = self.model.predict_generator(self.validation_data, verbose=1)
        predicted = np.argmax(predicted, axis=1)
        ground = self.validation_labels
        cm = sklearn.metrics.confusion_matrix(ground, predicted, labels=None, sample_weight=None)
        template = "{0:10}|{1:30}|{2:10}|{3:30}|{4:15}|{5:15}"
        print(template.format("", "", "", "Predicted", "", ""))
        print(template.format("", "", "Normal",
                              "No Lung Opacity / Not Normal", "Lung Opacity", "Total true"))
        print(template.format("", "="*28, "="*9, "="*28, "="*12, "="*12))
        print(template.format("", "Normal",
                              cm[0, 0], cm[0, 1], cm[0, 2], np.sum(cm[0, :])))
        print(template.format("True", "No Lung Opacity / Not Normal",
                              cm[1, 0], cm[1, 1], cm[1, 2], np.sum(cm[1, :])))
        print(template.format("", "Lung Opacity",
                              cm[2, 0], cm[2, 1], cm[2, 2], np.sum(cm[2, :])))
        print(template.format("", "Total predicted", np.sum(
            cm[:, 0]), np.sum(cm[:, 1]), np.sum(cm[:, 2]), ""))

@gin.configurable
def F1_score(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    F1 = numerator / denominator
    return tf.reduce_mean(F1)

@gin.configurable
def sensitivity(y_true, y_pred, smooth=1.):
    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
    return coef

@gin.configurable
def specificity(y_true, y_pred, smooth=1.):
    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
    return coef

@gin.configurable
def muilticlass_logloss(y_true, y_pred):
    return tf.losses.log_loss(y_true, y_pred)

@gin.configurable
def dice_coef_loss(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = -K.log(2. * intersection + smooth) + \
           K.log((K.sum(y_true_f) +
                  K.sum(y_pred_f) + smooth))
    return loss
