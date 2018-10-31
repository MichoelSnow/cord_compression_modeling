import gin
from infer import load_model
from vis_tools import ActMax
from metrics import F1_score, sensitivity, specificity, muilticlass_logloss


def run():
    gin.parse_config_file('config_vis.gin')
    # Load the model
    custom_losses = {'F1_score':F1_score, 'sensitivity':sensitivity, 'specificity':specificity,
                     'muilticlass_logloss':muilticlass_logloss}
    model = load_model(model_name='ResNet50.h5', model_path='/data/gferguso/cord_comp/', custom_objects=custom_losses)
    # Plot the activation maximization
    plot = ActMax(model=model)
    print(plot)

if __name__ == '__main__':
    run()
