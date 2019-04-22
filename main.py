import gin
from cv_framework import cv_scientist
import keras.backend as K

def run():
    # run gin-config
    #gin.parse_config_file('config_UNIT_TEST.gin')
    gin.parse_config_file('config_ARDS_Baseline.gin')
    # setup experiment
    experiment = cv_scientist.CompVisExperiment()

    model_dict = {'simple_CNN':['simple_low_lr', 'simple_norm_lr'],
                  'ResNet50':['resnet_low_lr', 'resnet_norm_lr'],
                  'InceptionResNetV2':['InResV2'],
                  'InceptionV3':['IncV2'],
                  'Multiscale_CNN':['Multi'],
                  'Xception':['Xcep'],
                  'DenseNet121':['DN121'],
                  'DenseNet169':['DN169'],
                  'DenseNet201':['DN201']}
    compiled_models = experiment.build_models(model_dict)
    train_list = list(compiled_models.keys())
    trained_models, model_table = experiment.train_models(train_list, compiled_models, save_figs=False,
                                                          print_class_rep=True, model_type='bin_classifier')
    print(model_table)
    model_table.to_csv('ards_baseline.csv')
    #print(f'sim_dog lr = {K.eval(compiled_models["sim_dog"].optimizer.lr)}')
    #print(f'sim_cat lr = {K.eval(compiled_models["sim_cat"].optimizer.lr)}')
    #print(f'res_dog lr = {K.eval(compiled_models["res_dog"].optimizer.lr)}')
    #print(f'res_cat lr = {K.eval(compiled_models["res_cat"].optimizer.lr)}')

if __name__ == '__main__':
    run()
