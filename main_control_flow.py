import gin
from cv_framework import control_flow
import keras.backend as K

def run():
    # run gin-config
    gin.parse_config_file('config_UNIT_TEST.gin')
    # setup experiment
    experiment = control_flow.CompVisExperiment()

    model_dict = {'simple_CNN':['sim_dog', 'sim_cat'], 'ResNet50':['res_dog', 'res_cat']}
    compiled_models = experiment.build_models(model_dict)
    train_list = list(compiled_models.keys())
    trained_models, model_table = experiment.train_models(train_list, compiled_models, save_figs=False,
                                                          print_class_rep=True, model_type='bin_classifier')
    print(model_table)
    print(f'sim_dog lr = {K.eval(compiled_models["sim_dog"].optimizer.lr)}')
    print(f'sim_cat lr = {K.eval(compiled_models["sim_cat"].optimizer.lr)}')
    print(f'res_dog lr = {K.eval(compiled_models["res_dog"].optimizer.lr)}')
    print(f'res_cat lr = {K.eval(compiled_models["res_cat"].optimizer.lr)}')

if __name__ == '__main__':
    run()
