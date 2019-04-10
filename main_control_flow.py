import gin
from cv_framework import control_flow

def run():
    # run gin-config
    gin.parse_config_file('config_UNIT_TEST.gin')

    # setup experiment
    experiment = control_flow.CompVisExperiment()

    model_list = ['simple_cnn', 'ResNet50']
    #model_list = ['simple_cnn']
    compiled_models = experiment.build_models(model_list)
    trained_models, model_table = experiment.train_models(model_list, compiled_models, save_figs=False,
                                                          print_class_rep=True)
    print(model_table)

if __name__ == '__main__':
    run()
