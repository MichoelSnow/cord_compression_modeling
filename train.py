import gin
import keras
import pandas as pd
import os
from metrics import ConfusionMatrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# integer encode

@gin.configurable
def get_val_data(img_dataframe='path_class.csv'):
    #!!!! SUPER HACKY workaround for  keras bug to get test labels
    test_dirs = ['./images/test/class_000_No_Lung_Opacity_and_Not_Normal','./images/test/class_001_Lung_Opacity',
                 './images/test/class_002_Normal']
    test_files_list = []
    test_data_numpy = []
    for dir in test_dirs:
        test_files_list = test_files_list + [os.path.join(dir, file) for file in os.listdir(dir)]
    test_file_df = pd.DataFrame({'full_image_paths':test_files_list})
    test_file_df['image_paths'] = test_file_df['full_image_paths'].str.replace(r'./images/test/\w+/','')
    labels_df = pd.read_csv(img_dataframe)
    labels_df['image_paths'] = labels_df['image_paths'].str.replace('/data/gferguso/datasets/RSNA_data/TRAIN_data/','',regex=False)
    test_df = pd.merge(test_file_df, labels_df, how='inner', on='image_paths')
    test_data_labels = test_df['class'].tolist()
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = label_encoder.fit_transform(np.array(test_data_labels))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded_labels

@gin.configurable
def set_keras_callbacks(calls=None, batch_size=None, gen=None, checkpoint_name='test.h5'):
    if not calls:
        added_calls = []
    else:
        added_calls = calls
    val_lb = get_val_data()
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor="val_loss",
                                                          verbose=1, save_best_only=True, period=1)
    tb_callback = keras.callbacks.TensorBoard(log_dir='./logs')
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    rop_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0,
                                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    confusion_callback = ConfusionMatrix(val_gen=gen, val_labels=val_lb, batch_size=batch_size)
    base_calls = [tb_callback, checkpoint_callback, earlystopping_callback, rop_callback,confusion_callback]
    fin_calls = base_calls + added_calls
    return fin_calls


@gin.configurable
def call_fit_gen(model=None, gen=None, steps_per_epoch=None, epochs=100, validation_data=None, validation_steps=None,
                 class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False):
    callbacks = set_keras_callbacks(gen=validation_data)
    history = model.fit_generator(generator=gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=validation_data,
                                  validation_steps=validation_steps,
                                  class_weight=class_weight,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)
    return history

@gin.configurable
def save_model(model=None, model_name='test.h5'):
    model.save(model_name)

