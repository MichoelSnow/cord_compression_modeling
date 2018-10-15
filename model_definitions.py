import keras
import gin

@gin.configurable
def set_input_output(input_shape=(None, None, None), output_shape=(None)):
    image_size=(input_shape[0], input_shape[1])
    in_shape = input_shape
    out_shape = output_shape[0]
    return image_size, in_shape, out_shape

@gin.configurable
def simple_CNN(input_values=(None, None, 1), output=3):
    ''' A simple CNN for testing'''
    input     = keras.layers.Input(shape=(input_values[0], input_values[1], input_values[2]), name='scnn_input')
    conv1     = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(input)
    max_pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2     = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool2)
    flatten   = keras.layers.Flatten()(conv3)
    dense1    = keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer='he_uniform')(flatten)
    pred      = keras.layers.Dense(output, activation='softmax', use_bias=True, kernel_initializer='he_uniform')(dense1)
    model = keras.Model(inputs=input, outputs=pred)
    return model

@gin.configurable
def simple_sep_CNN(input_values=(None, None, 1), output=3):
    ''' A simple CNN for testing'''
    input     = keras.layers.Input(shape=(input_values[0], input_values[1], input_values[2]), name='scnn_input')
    conv1     = keras.layers.SeparableConv2D(filters=128, kernel_size=(5, 5), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(input)
    max_pool1 = keras.layers.MaxPooling2D(pool_size=(3,3))(conv1)
    conv2     = keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool2)
    flatten   = keras.layers.Flatten()(conv3)
    dense1    = keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer='he_uniform')(flatten)
    pred      = keras.layers.Dense(output, activation='softmax', use_bias=True, kernel_initializer='he_uniform')(dense1)
    model = keras.Model(inputs=input, outputs=pred)
    return model

@gin.configurable
def deep_sep_CNN(input_values=(None, None, 1), output=3):
    ''' A simple CNN for testing'''
    input     = keras.layers.Input(shape=(input_values[0], input_values[1], input_values[2]), name='scnn_input')
    conv1     = keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(input)
    max_pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2     = keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool2)
    conv4     = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(conv3)
    conv5     = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(conv4)
    conv6     = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(conv5)
    flatten   = keras.layers.Flatten()(conv6)
    dense1    = keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer='he_uniform')(flatten)
    pred      = keras.layers.Dense(output, activation='softmax', use_bias=True, kernel_initializer='he_uniform')(dense1)
    model = keras.Model(inputs=input, outputs=pred)
    return model

@gin.configurable
def multiscale_network():
    pass
    # # main CNN model - CNN1
    # main_model = Sequential()
    # main_model.add(Convolution2D(32, 3, 3, input_shape=(3, 224, 224)))
    # main_model.add(Activation('relu'))
    # main_model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # main_model.add(Convolution2D(32, 3, 3))
    # main_model.add(Activation('relu'))
    # main_model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # main_model.add(Convolution2D(64, 3, 3))
    # main_model.add(Activation('relu'))
    # main_model.add(MaxPooling2D(pool_size=(2, 2)))  # the main_model so far outputs 3D feature maps (height, width, features)
    #
    # main_model.add(Flatten())
    #
    # # lower features model - CNN2
    # lower_model1 = Sequential()
    # lower_model1.add(Convolution2D(32, 3, 3, input_shape=(3, 224, 224)))
    # lower_model1.add(Activation('relu'))
    # lower_model1.add(MaxPooling2D(pool_size=(2, 2)))
    # lower_model1.add(Flatten())
    #
    # # lower features model - CNN3
    # lower_model2 = Sequential()
    # lower_model2.add(Convolution2D(32, 3, 3, input_shape=(3, 224, 224)))
    # lower_model2.add(Activation('relu'))
    # lower_model2.add(MaxPooling2D(pool_size=(2, 2)))
    # lower_model2.add(Flatten())
    #
    # # merged model
    # merged_model = Merge([main_model, lower_model1, lower_model2], mode='concat')
    #
    # final_model = Sequential()
    # final_model.add(merged_model)
    # final_model.add(Dense(64))
    # final_model.add(Activation('relu'))
    # final_model.add(Dropout(0.5))
    # final_model.add(Dense(1))
    # final_model.add(Activation('sigmoid'))
    # final_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # print
    # 'About to start training merged CNN'
    # train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224), batch_size=32,
    #                                                     class_mode='binary')
    #
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(args.test_images, target_size=(224, 224), batch_size=32,
    #                                                   class_mode='binary')
    #
    # final_train_generator = zip(train_generator, train_generator, train_generator)
    # final_test_generator = zip(test_generator, test_generator, test_generator)
    # final_model.fit_generator(final_train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch,
    #                           validation_data=final_test_generator, nb_val_samples=nb_validation_samples)

@gin.configurable
def dense_net169(input_values=(None, None, 3), classes=3):
    """ A DenseNet169 Model from Keras."""
    DenseNet169 = keras.applications.densenet.DenseNet169(include_top=False,
                                                          weights=None,
                                                          input_tensor=None,
                                                          input_shape=(input_values[0],input_values[1],input_values[2]),
                                                          pooling='max',
                                                          classes=classes)
    last_layer = DenseNet169.output
    preds = keras.layers.Dense(classes, activation='sigmoid')(last_layer)
    model = keras.Model(DenseNet169.input, preds)
    return model

def _resnet_layer(inputs, fmaps, name):
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


