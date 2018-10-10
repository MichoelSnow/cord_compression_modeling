import gin

@gin.configurable
def call_fit_gen(model=None, gen=None, steps_per_epoch=None, epochs=100, callbacks=None, validation_data=None,
                validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False):
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


# tb_callback = K.callbacks.TensorBoard(log_dir="./logs")
# model_callback = K.callbacks.ModelCheckpoint(
#     "../models/baseline_classifier.h5",
#     monitor="val_loss", verbose=1,
#     save_best_only=True)
# early_callback = K.callbacks.EarlyStopping(monitor="val_loss",
#                                            patience=3, verbose=1)
#
# confusion_callback = ConfusionMatrix(imgs_test, labels_test, 3)
#
# callbacks = [tb_callback, model_callback, early_callback, confusion_callback]
#
# model = simple_lenet()
#
# model.compile(loss="categorical_crossentropy",
#               optimizer=K.optimizers.Adam(lr=0.000001),
#               metrics=["accuracy"])
#
# model.summary()
#
# class_weights = {0: 0.1, 1: 0.1, 2: 0.8}
# print("Class weights = {}".format(class_weights))
#
# model.fit(imgs_train, labels_train,
#           epochs=epochs,
#           batch_size=batch_size,
#           verbose=1,
#           #class_weight=class_weights,
#           validation_data=(imgs_test, labels_test),
#           callbacks=callbacks)
