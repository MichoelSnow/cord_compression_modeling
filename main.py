import time
import os
import pandas as pd
import tensorflow as tf
import gin
import shutil
from sklearn.model_selection import train_test_split
import keras

@gin.configurable
class DataPrepper:
    def __init__(self, img_dataframe=None, remake_dirs=False, label_col=None, train=0.60):
        self.data_dir = os.path.join(os.getcwd(), 'images')
        assert isinstance(img_dataframe, pd.DataFrame), f'{img_dataframe} must be a Dataframe.'
        self.data = img_dataframe.drop(label_col)
        self.remake_dirs = remake_dirs
        self.label_col = label_col
        self.train = train

    def _make_dirs(self, paths):
        assert isinstance(paths, list), 'Only make directories from lists, not {}'.format(type(paths))
        for path in paths:
            if self.remake_dirs:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path)
            else:
                try:
                    os.makedirs(path)
                except OSError:
                    print(f'Directory {path} already exists. To remake set remake_dirs=True.')

    def label_names(self):
        return self.label_col.unique()

    def _train_test_val(self):
        img_train, img_test_val, label_train, label_test_val  = train_test_split(self.data, self.labels,
                                                                                 train_size=self.train,
                                                                                     test_size=1-self.train,
                                                                                 random_state=42)
        img_test, img_val, label_test, label_val = train_test_split(img_test_val, label_test_val,
                                                                    train_size=0.5, test_size=1 - self.train,
                                                                    random_state=42)
        data_dict = {'img_train':img_train,'label_train':label_train,'img_test':img_test,'label_test':label_test,
                     'img_val':img_val,'label_val':label_val}
        return data_dict

    def build_dataset(self, data_dir=None):
        if not data_dir:
            data_dir=self.data_dir
        train_path = os.path.join(data_dir, 'train')
        test_path  = os.path.join(data_dir, 'test')
        valid_path = os.path.join(data_dir, 'validate')
        paths = [train_path, test_path, valid_path]
        self._make_dirs(paths)
        data_dict = self._train_test_val()
        label_names = self.label_names()
        for i, label in enumerate(label_names):
            dir_name = f'Class_{i:03d}_{label}'
            train_path = os.path.join(train_path, dir_name)
            test_path  = os.path.join(test_path, dir_name)
            valid_path = os.path.join(valid_path, dir_name)
            path_classes = [train_path, test_path, valid_path]
            if not os.listdir(train_path):
                self._make_dirs(path_classes)
                train_files = data_dict['img_train'][data_dict['label_train'].label == label].to_list()
                for file in train_files:
                    file_name = os.path.join(data_dir, file)
                    shutil.copy(file_name, train_path)
                test_files  = data_dict['img_test'][data_dict['label_test'].label == label].to_list()
                for file in test_files:
                    file_name = os.path.join(data_dir, file)
                    shutil.copy(file_name, test_path)
                valid_files = data_dict['img_val'][data_dict['label_val'].label == label].to_list()
                for file in valid_files:
                    file_name = os.path.join(data_dir, file)
                    shutil.copy(file_name, valid_path)
            else:
                print('Directories exist and are not empty, please delete to remake.')




#LEARNING_RATE = 0.0001
# DECAY_FACTOR = 10 #learning rate decayed when valid. loss plateaus after an epoch
#ADAM_B1 = 0.9 #adam optimizer default beta_1 value (Kingma & Ba, 2014)
#ADAM_B2 = 0.999 #adam optimizer default beta_2 value (Kingma & Ba, 2014)

#MAX_ROTAT_DEGREES = 30 #up to 30 degrees img rotation.
#MIN_ROTAT_DEGREES = 0

#TB_LOG_DIR = "../logs/"
#CHECKPOINT_FILENAME = "../checkpoints/Baseline_{}.hdf5".format(time.strftime("%Y%m%d_%H%M%S"))# Save Keras model to this file
#MODEL_FILENAME = "../models/Baseline_model.h5"

def main():
    print("Building Train/Validation Dataset Objects")
    data_prep = DataPrepper()
    data_prep.build_dataset()

#
#
#     #Instantiate MODEL:
#     model = lenet(IMG_RESIZE_X,IMG_RESIZE_Y,CHANNELS) #tf.dataset.shape???
#
#     # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
#     optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
#             beta1=ADAM_B1,
#             beta2=ADAM_B2)
#
#     optimizer_keras = tf.keras.optimizers.Adam(lr=LEARNING_RATE,
#             beta_1=ADAM_B1,
#             beta_2=ADAM_B2,
#             decay=0.10)
#
#     # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
#     checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
#             monitor="val_loss",
#             verbose=1,
#             save_best_only=True)
#
#     # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
#     #TODO custom tensorboard log file names..... or write to unqiue dir as a sub dir in the logs file...
#     tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
#             # histogram_freq=1, #this screwed us over... caused tensorboard callback to fail.. why??? DEBUG !!!!!!
#             # batch_size=BATCH_SIZE, # and take this out... and boom.. histogam frequency works. sob
#             write_graph=True,
#             write_grads=False,
#             write_images=True)
#
#     print("Compiling Model!")
#     model.compile(optimizer=optimizer_keras,
#             loss='binary_crossentropy',
#             metrics=['accuracy'])
#
#     print("Beginning to Train Model")
#     model.fit(train_dataset,
#             epochs=EPOCHS,
#             steps_per_epoch=(len(train_labels)//BATCH_SIZE), #36808 train number
#             verbose=1,
#             validation_data=valid_dataset,
#             validation_steps= (len(valid_labels)//BATCH_SIZE),  #3197 validation number
#             callbacks=[checkpointer,tensorboard])  #https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
#
#
#     # Save entire model to a HDF5 file
#     model.save(MODEL_FILENAME)
#     # # Recreate the exact same model, including weights and optimizer.
#     # model = keras.models.load_model('my_model.h5')


if __name__ == '__main__':
    run(data)
