import gin
import math
import os
import pandas as pd
import numpy as np
from cv_framework.data_access.data_prep import FilePrep
from cv_framework.model_definitions.model_utils import set_input_output
from cv_framework.training.model_comp import comp_model
from cv_framework.data_access.generators import set_dir_flow_generator
from cv_framework.training.train import save_model, call_fit_gen
from cv_framework.model_definitions import model_dict
from cv_framework.diagnostics import basic_diagnostics

@gin.configurable
class CompVisExperiment:

    def __init__(self, base_directory=None, image_directory=None, experiment_name=None, labels_csv=None,
                 file_name_column='file_name', labels_column='class',  use_symlinks=False):
        exp_dir = os.path.join(base_directory, str(experiment_name))
        if not os.path.exists(exp_dir):
            try:
                os.makedirs(exp_dir)
            except Exception as e:
                print(f'Could not make experimental directory: {e}')


        print("Building Train/Test/Validation data directories.")
        if not image_directory:
            image_directory = os.path.join(exp_dir, 'images')

        labels_csv_path = os.path.join(base_directory, labels_csv)
        file_prep = FilePrep(exp_directory=exp_dir, image_directory=image_directory, labels_csv_path=labels_csv_path, file_name_column=file_name_column,
                             labels_column=labels_column, use_symlinks=use_symlinks)
        file_prep.create_modeling_dataset()

        self.unique_class_labels = file_prep.label_names

        train_dir = os.path.join(exp_dir + '/train')
        test_dir  = os.path.join(exp_dir + '/test')

        self.image_size, self.in_shape, self.out_shape = set_input_output()
        self.train_gen = set_dir_flow_generator(dir=train_dir, shuffle=True, image_size=self.image_size)
        self.test_gen = set_dir_flow_generator(dir=test_dir, shuffle=False, image_size=self.image_size)

    def available_models(self):
        print(list(model_dict.models_dict.keys()))

    @gin.configurable
    def build_models(self, model_list, summary=True):
        compiled_models = {}
        for model in model_list:
            cnn_model = model_dict.models_dict[model](input_shape=self.in_shape, classes=self.out_shape)
            if summary:
                cnn_model.summary()
            compiled_models[model] = comp_model(model=cnn_model)
        return compiled_models

    @gin.configurable
    def train_models(self, train_list, compiled_models, model_type='bin_classifier', save_figs=False,
                     print_class_rep=True):
        history_dict = {}
        score_dict = {}
        for model in train_list:
            save_name = str(model) + ('.h5')
            history = call_fit_gen(
                model=compiled_models[model],
                gen=self.train_gen,
                validation_data=self.test_gen)
            save_model(model=compiled_models[model], model_name=save_name)
            history_dict[model] = history.history
            self.test_gen.reset()
            preds = compiled_models[model].predict_generator(
                self.test_gen,  verbose=1, steps=math.ceil(len(self.test_gen.classes)/self.test_gen.batch_size)
            )
            score_dict[model] = self.score_models(preds, model, history=history_dict[model], save_figs=save_figs,
                                                  model_type=model_type, print_class_rep=print_class_rep)

        model_table = pd.DataFrame(score_dict).transpose().reset_index().rename(mapper={'index':'Model_Name'}, axis=1)
        return compiled_models, model_table

    def score_models(self, preds, model, history=None, save_figs=False, model_type=None, print_class_rep=None):
        if model_type == 'bin_classifier':
            return self.score_binary_classifiers(preds, model, history, save_figs, print_class_rep)

    @gin.configurable
    def score_binary_classifiers(self, preds, model, history, save_figs, print_class_rep):
        model = str(model)
        sens, spec, roc_auc, class_rep, TP, TN, FP, FN, PPV, NPV, FPR, FNR = basic_diagnostics.binary_metrics(
            self.test_gen.classes, preds, history=history, save_figs=save_figs, class_names=self.unique_class_labels,
            model_name=model
        )
        model_scores = {'Sensitivity':sens, 'Specificity':spec, 'ROC_AUC_SCORE':roc_auc, 'True_Positives':TP,
                        'True_Negatives':TN, 'False_Positives':FP, 'False_Negatives':FN,
                        'Positive_Predictive_Value':PPV, 'Negative_Predictive_Value':NPV, 'False_Positive_Rate':FPR,
                        'False_Negative_Rate':FNR}

        if print_class_rep:
            print(class_rep)

        return model_scores




