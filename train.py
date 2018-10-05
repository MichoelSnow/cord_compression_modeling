import tensorflow as tf
import keras as K
import numpy as np
import sklearn
import os

batch_size = 512
epochs = 100
resize_height = 32 # 512  # Resize images to this height
resize_width = 32 #512   # Resize images to this width

data_path = "../../rsna_data_numpy/"

imgs_train = np.load(os.path.join(data_path, "imgs_train.npy"),
                     mmap_mode="r", allow_pickle=False)
imgs_test = np.load(os.path.join(data_path, "imgs_test.npy"),
                    mmap_mode="r", allow_pickle=False)

labels_train = np.load(os.path.join(data_path, "labels_train.npy"),
                       mmap_mode="r", allow_pickle=False)
labels_test = np.load(os.path.join(data_path, "labels_test.npy"),
                      mmap_mode="r", allow_pickle=False)



def resize_normalize(image, resize_height, resize_width):
    """
    Resize images on the graph
    """
    from tensorflow.image import resize_images

    resized = resize_images(image, (resize_height, resize_width))

    return resized


tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
model_callback = K.callbacks.ModelCheckpoint(
    "../models/baseline_classifier.h5",
    monitor="val_loss", verbose=1,
    save_best_only=True)
early_callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=3, verbose=1)

confusion_callback = ConfusionMatrix(imgs_test, labels_test, 3)

callbacks = [tb_callback, model_callback, early_callback, confusion_callback]

model = simple_lenet()

model.compile(loss="categorical_crossentropy",
              optimizer=K.optimizers.Adam(lr=0.000001),
              metrics=["accuracy"])

model.summary()

class_weights = {0: 0.1, 1: 0.1, 2: 0.8}
print("Class weights = {}".format(class_weights))

model.fit(imgs_train, labels_train,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          #class_weight=class_weights,
          validation_data=(imgs_test, labels_test),
          callbacks=callbacks)
