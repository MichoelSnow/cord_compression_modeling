import tensorflow as tf
from tensorflow import keras
import keras

def dense_net169(img_x,img_y,channels,label_shape,classes=2):
    """ todo """
    print("Downloading DenseNet PreTrained Weights... Might take ~0:30 seconds")
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


def simple_lenet(dropout_rate=0.5):

    inputs = K.layers.Input(shape=(None, None, 1), name="Images")

    inputR = K.layers.Lambda(resize_normalize,
                             input_shape=(None, None, 1),
                             output_shape=(resize_height, resize_width,1),
                             arguments={"resize_height":resize_height,
                                        "resize_width": resize_width})(inputs)


    conv = K.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(inputR)

    #conv = K.layers.BatchNormalization()(conv)

    conv = K.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(conv)

    #conv = K.layers.BatchNormalization()(conv)

    pool = K.layers.MaxPooling2D(pool_size=(2, 2))(conv)

    conv = K.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(pool)

    #conv = K.layers.BatchNormalization()(conv)

    conv = K.layers.Conv2D(filters=128,
                           kernel_size=(3, 3),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(conv)

    #conv = K.layers.BatchNormalization()(conv)

    pool = K.layers.MaxPooling2D(pool_size=(2, 2))(conv)

    flat = K.layers.Flatten()(pool)

    dense1 = K.layers.Dense(256, activation="relu")(flat)
    dropout = K.layers.Dropout(dropout_rate)(dense1)
    dense2 = K.layers.Dense(128, activation="relu")(dense1)
    dropout = K.layers.Dropout(dropout_rate)(dense2)

    prediction = K.layers.Dense(3, activation="softmax")(dropout)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    return model


def resnet_layer(inputs, fmaps, name):
    """
    Residual layer block
    """

    conv1 = K.layers.Conv2D(name=name+"a", filters=fmaps*2,
                            kernel_size=(1, 1), activation="linear",
                            padding="same",
                            kernel_initializer="he_uniform")(inputs)
    conv1b = K.layers.BatchNormalization()(conv1)

    conv1b = K.layers.Conv2D(name=name+"b", filters=fmaps,
                             kernel_size=(3, 3), activation="linear",
                             padding="same",
                             kernel_initializer="he_uniform")(conv1b)
    conv1b = K.layers.BatchNormalization()(conv1b)
    conv1b = K.layers.Activation("relu")(conv1b)

    conv1b = K.layers.Conv2D(name=name+"c", filters=fmaps*2,
                             kernel_size=(1, 1), activation="linear",
                             padding="same",
                             kernel_initializer="he_uniform")(conv1b)

    conv_add = K.layers.Add(name=name+"_add")([conv1, conv1b])
    conv_add = K.layers.BatchNormalization()(conv_add)

    pool = K.layers.MaxPooling2D(name=name+"_pool", pool_size=(2, 2))(conv_add)

    return pool


def resnet(dropout=0.5):

    inputs = K.layers.Input(shape=(None, None, 1), name="Images")

    inputR = K.layers.Lambda(resize_normalize,
                             input_shape=(None, None, 1),
                             output_shape=(resize_height, resize_width,1),
                             arguments={"resize_height":resize_height,
                                        "resize_width": resize_width})(inputs)

    pool1 = resnet_layer(inputR, 16, "conv1")
    pool2 = resnet_layer(pool1, 32, "conv2")
    pool3 = resnet_layer(pool2, 64, "conv3")
    pool4 = resnet_layer(pool3, 128, "conv4")

    pool = pool2

    conv = K.layers.Conv2D(name="NiN2", filters=64,
                           kernel_size=(1, 1),
                           activation="relu",
                           padding="valid",
                           kernel_initializer="he_uniform")(pool)
    conv = K.layers.Dropout(dropout)(conv)

    conv = K.layers.Conv2D(name="1x1", filters=3,
                           kernel_size=(1, 1),
                           activation="linear",
                           padding="same",
                           kernel_initializer="he_uniform")(conv)

    gap1 = K.layers.GlobalAveragePooling2D()(conv)

    prediction = K.layers.Activation(activation="softmax")(gap1)

    model = K.models.Model(inputs=[inputs], outputs=[prediction])

    return model
