import keras

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
