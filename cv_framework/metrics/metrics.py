import keras
from keras.callbacks import Callback
from keras import backend as K
import numpy as np
import sklearn
import math

class Summary_metrics(Callback):
    def __init__(self, val_gen):
        super().__init__()
        #val_gen.reset()
        self.validation_gen = val_gen
        self.validation_labels = val_gen.classes
        self.val_batch = val_gen.batch_size
        self.class_indices = val_gen.class_indices

    def on_epoch_end(self, epoch, logs=None):
        print("Calculating confusion matrix")
        self.validation_gen.reset()
        #print(vars(self.validation_gen))
        num_val_gen_steps = math.ceil(len(self.validation_labels)/self.val_batch)
        predicted = self.model.predict_generator(self.validation_gen, verbose=1, steps=num_val_gen_steps)
        predicted = np.argmax(predicted, axis=1)
        ground = self.validation_labels
        #print(len(ground))
        #print(len(predicted))
        cm = sklearn.metrics.confusion_matrix(ground, predicted, labels=None, sample_weight=None)
        print(cm)
        report = sklearn.metrics.classification_report(ground, predicted)
        print('Classification Report: \n{}'.format(report))
        # labels = list(set(self.validation_labels))
        # template = '{0:10}|{1:30}|{2:10}|{3:30}|{4:15}|{5:15}'
        # print(template.format('', '', '', 'Predicted', '', ''))
        # print(template.format('', '', labels[0], labels[1], labels[2], 'Total true'))
        # print(template.format('', '='*28, '='*9, '='*28, '='*12, '='*12))
        # print(template.format('', labels[0], cm[0, 0], cm[0, 1], cm[0, 2], np.sum(cm[0, :])))
        # print(template.format('True',labels[1], cm[1, 0], cm[1, 1], cm[1, 2], np.sum(cm[1, :])))
        # print(template.format('', labels[2], cm[2, 0], cm[2, 1], cm[2, 2], np.sum(cm[2, :])))
        # print(template.format('', 'Total predicted', np.sum(cm[:, 0]), np.sum(cm[:, 1]), np.sum(cm[:, 2]), ''))

def muilticlass_logloss(y_true, y_pred):
   return K.tf.losses.log_loss(y_true, y_pred)
