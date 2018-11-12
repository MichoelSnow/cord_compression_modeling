import pandas as pd
import gin
from data_prep import FilePrep
from model_definitions.model_utils import set_input_output
from model_comp import comp_model
from generators import set_dir_flow_generator
from train import save_model, call_fit_gen
from model_definitions.simple_models import simple_sep_CNN, simple_CNN
from model_definitions.ResNet50 import (ResNet50)
from model_definitions.InceptionResNetV2 import InceptionResNetV2
from model_definitions.InceptionV3 import InceptionV3
from model_definitions.Multiscale_CNN import Multiscale_CNN
from model_definitions.NASNnet import NASNet
from model_definitions.Xception import Xception
from model_definitions.DenseNet import DenseNet121,DenseNet169, DenseNet201, DenseNetCustom


def run():
    gin.parse_config_file('config.gin')
    # Make the train/test/validation directory structure, perform a test-train_split
    print("Building Train/Test/Validation data directories.")
    file_prep = FilePrep()
    file_prep.build_dataset()
    image_size, in_shape, out_shape = set_input_output()

    ####### MultisclaeCNN
    print('Fitting Multiscale CNN')

    # Add a model
    cnn_model = Multiscale_CNN(input_shape=in_shape, classes=out_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    # Build the file generator
    train_gen = set_dir_flow_generator(dir='/home/cordcomp/cord_data/images/train', image_size=image_size)
    test_gen = set_dir_flow_generator(dir='/home/cordcomp/cord_data/images/test', image_size=image_size)
    # Fit the model
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='Multiscale_CNN.h5')
    print(history.history)

    ####### ResNet50
    print('Fitting ResNet50')

    cnn_model = ResNet50(input_shape=in_shape, classes=out_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='ResNet50.h5')
    print(history.history)


    ####### DenseNet121
    print('Fitting DenseNet121')

    cnn_model = DenseNet121(input_shape=in_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='DenseNet121.h5')
    print(history.history)


    ####### DenseNet169
    print('Fitting DenseNet169')

    cnn_model = DenseNet169(input_shape=in_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='DenseNet169.h5')
    print(history.history)


    ####### DenseNet201
    print('Fitting DenseNet201')

    cnn_model = DenseNet201(input_shape=in_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='DenseNet201.h5')
    print(history.history)

    ####### InceptionV3
    print('Fitting InceptionV3')

    cnn_model = InceptionV3(input_shape=in_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='InceptionV3.h5')
    print(history.history)

    ####### InceptionResNetV2
    print('Fitting InceptionResNetV2')

    cnn_model = InceptionResNetV2(input_shape=in_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='InceptionResNetV2.h5')
    print(history.history)

    ####### Xception
    print('Fitting Xception')

    cnn_model = Xception(input_shape=in_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)
    save_model(model=cnn_model, model_name='Xception.h5')
    print(history.history)

    ####### NASNet
    print('Fitting NasNet')
    cnn_model = NASNet(input_shape=in_shape)
    comp_model(model=cnn_model)

if __name__ == '__main__':
    run()
