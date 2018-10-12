import pandas as pd
import gin
from data_prep import FilePrep
from model_definitions import simple_CNN, dense_net169, simple_sep_CNN
from model_comp import set_adam, set_loss, set_metrics
from generators import set_dir_flow_generator
from train import call_fit_gen, save_model, get_val_data

def run(data):
    gin.parse_config_file('config.gin')
    # Make the train/test/validation directory structure, perform a test-train_split
    print("Building Train/Test/Validation data directories.")
    #file_prep = FilePrep(img_dataframe=data)
    #file_prep.build_dataset()

    # Add a model
    #cnn_model = simple_CNN()
    #cnn_model = dense_net169()
    cnn_model = simple_sep_CNN()

    # set compile value and compile model
    k_opt = set_adam()
    k_loss = set_loss()
    k_metrics = set_metrics()
    cnn_model.compile(optimizer=k_opt,
                             loss=k_loss,
                             metrics=k_metrics)

    #cnn_model.summary()

    # Build the file generator
    train_gen = set_dir_flow_generator(dir = '/data/gferguso/cord_comp/images/train')
    test_gen = set_dir_flow_generator(dir = '/data/gferguso/cord_comp/images/test')

    #get_val_data()

    # Fit the model
    history = call_fit_gen(model=cnn_model,
                           gen=train_gen,
                           validation_data=test_gen)

    #save_model(model=cnn_model, model_name='test.h5')

    #print(history.history)


data = pd.read_csv('path_class.csv')
if __name__ == '__main__':
    run(data)
