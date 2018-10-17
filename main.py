import pandas as pd
import gin
from data_prep import FilePrep
from model_definitions import set_input_output, simple_CNN, simple_sep_CNN, deep_sep_CNN
from model_comp import comp_model
from generators import set_dir_flow_generator
from train import call_fit_gen, save_model

def run():
    gin.parse_config_file('config.gin')
    # Make the train/test/validation directory structure, perform a test-train_split
    print("Building Train/Test/Validation data directories.")
    file_prep = FilePrep()
    file_prep.build_dataset()
    image_size, in_shape, out_shape = set_input_output()

    # Add a model
    #cnn_model = simple_CNN()
    #cnn_model = dense_net169()
    cnn_model = deep_sep_CNN(input_values=in_shape, output=out_shape)
    comp_model(model=cnn_model)
    cnn_model.summary()

    # Build the file generator
    train_gen = set_dir_flow_generator(dir='/data/gferguso/cord_comp/images/train', image_size=image_size)
    test_gen = set_dir_flow_generator(dir='/data/gferguso/cord_comp/images/test', image_size=image_size)

    # # Fit the model
    # history = call_fit_gen(model=cnn_model,
    #                        gen=train_gen,
    #                        validation_data=test_gen)

    #save_model(model=cnn_model, model_name='test.h5')

    #print(history.history)

data = pd.read_csv('path_class.csv')
if __name__ == '__main__':
    run()
