import keras
import gin

@gin.configurable
def simple_CNN(input_values=(None, None, 1), output=3):
    ''' A simple CNN for testing'''
    input     = keras.layers.Input(shape=(input_values[0], input_values[1], input_values[2]), name='scnn_input')
    conv1     = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(input)
    max_pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2     = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(1,1))(conv2)
    conv3     = keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool2)
    flatten   = keras.layers.Flatten()(conv3)
    dense1    = keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer='he_uniform')(flatten)
    pred      = keras.layers.Dense(output, activation='softmax', use_bias=True, kernel_initializer='he_uniform')(dense1)
    model = keras.Model(inputs=input, outputs=pred)
    return model


def dense_net169(self,img_x,img_y,channels,label_shape,classes=3):
    """ A DenseNet169 Model from Keras."""
    DenseNet169 = keras.applications.densenet.DenseNet169(include_top=False,
                                                          weights='imagenet',
                                                          input_tensor=None,
                                                          input_shape=(img_x, img_y, channels),
                                                          pooling='max',
                                                          classes=classes)
    last_layer = DenseNet169.output
    preds = keras.layers.Dense(label_shape[-1], activation='sigmoid')(last_layer)
    model = keras.Model(DenseNet169.input, preds)
    return model

def _resnet_layer(self, inputs, fmaps, name):
    """
    Residual layer block
    """
    conv1 = keras.layers.Conv2D(name=name+"a", filters=fmaps*2,
                                kernel_size=(1, 1), activation="linear",
                                padding="same",
                                kernel_initializer="he_uniform")(inputs)
    conv1b = keras.layers.BatchNormalization()(conv1)

    conv1b = keras.layers.Conv2D(name=name+"b", filters=fmaps,
                             kernel_size=(3, 3), activation="linear",
                             padding="same",
                             kernel_initializer="he_uniform")(conv1b)
    conv1b = keras.layers.BatchNormalization()(conv1b)
    conv1b = keras.layers.Activation("relu")(conv1b)

    conv1b = keras.layers.Conv2D(name=name+"c", filters=fmaps*2,
                             kernel_size=(1, 1), activation="linear",
                             padding="same",
                             kernel_initializer="he_uniform")(conv1b)

    conv_add = keras.layers.Add(name=name+"_add")([conv1, conv1b])
    conv_add = keras.layers.BatchNormalization()(conv_add)

    pool = keras.layers.MaxPooling2D(name=name+"_pool", pool_size=(2, 2))(conv_add)

    return pool


def resnet(self, dropout=0.5):

    inputs = keras.layers.Input(shape=(None, None, 1), name="Images")

    inputR = keras.layers.Lambda(resize_normalize,
                             input_shape=(None, None, 1),
                             output_shape=(resize_height, resize_width,1),
                             arguments={"resize_height":resize_height,
                                        "resize_width": resize_width})(inputs)

    pool1 = self._resnet_layer(inputR, 16, "conv1")
    pool2 = self._resnet_layer(pool1, 32, "conv2")
    pool3 = self._resnet_layer(pool2, 64, "conv3")
    pool4 = self._resnet_layer(pool3, 128, "conv4")

    pool = pool2

    conv = keras.layers.Conv2D(name="NiN2", filters=64,
                           kernel_size=(1, 1),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(pool)
    conv = keras.layers.Dropout(dropout)(conv)

    conv = keras.layers.Conv2D(name="1x1", filters=3,
                           kernel_size=(1, 1),
                           activation="linear",
                           padding="same",
                           kernel_initializer="he_uniform")(conv)

    gap1 = keras.layers.GlobalAveragePooling2D()(conv)

    prediction = keras.layers.Activation(activation="softmax")(gap1)

    model = keras.models.Model(inputs=[inputs], outputs=[prediction])

    return model

# # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
#         beta1=ADAM_B1,
#         beta2=ADAM_B2)
#
# optimizer_keras = tf.keras.optimizers.Adam(lr=LEARNING_RATE,
#         beta_1=ADAM_B1,
#         beta_2=ADAM_B2,
#         decay=0.10)
#
# # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
# checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILENAME,
#         monitor="val_loss",
#         verbose=1,
#         save_best_only=True)
#
# # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
# #TODO custom tensorboard log file names..... or write to unqiue dir as a sub dir in the logs file...
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
#         # histogram_freq=1, #this screwed us over... caused tensorboard callback to fail.. why??? DEBUG !!!!!!
#         # batch_size=BATCH_SIZE, # and take this out... and boom.. histogam frequency works. sob
#         write_graph=True,
#         write_grads=False,
#         write_images=True)
#
# print("Compiling Model!")
# model.compile(optimizer=optimizer_keras,
#         loss='binary_crossentropy',
#         metrics=['accuracy'])
#
# print("Beginning to Train Model")
# model.fit(train_dataset,
#         epochs=EPOCHS,
#         steps_per_epoch=(len(train_labels)//BATCH_SIZE), #36808 train number
#         verbose=1,
#         validation_data=valid_dataset,
#         validation_steps= (len(valid_labels)//BATCH_SIZE),  #3197 validation number
#         callbacks=[checkpointer,tensorboard])  #https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
#
#
# # Save entire model to a HDF5 file
# model.save(MODEL_FILENAME)
# # # Recreate the exact same model, including weights and optimizer.
# # model = keras.models.load_model('my_model.h5')
