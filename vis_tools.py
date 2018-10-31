import keras
#from vis.utils import utils
#from vis.visualization import visualize_activation
import vis
import gin
import matplotlib.pylab as plt

@gin.configurable
def ActMax(model=None, layer_name=None, filter_index=0, backprop_modifier=None, grad_modifier=None,
           act_max_weight=None, lp_norm_weight=0., tv_weight=0., input_range=None, verbose=False,
           opt_max_iter=200, opt_input_mods=None, opt_callbacks=None):

    layer_idx = utils.find_layer_idx(model, layer_name)


    layer_idx = vis.utils.find_layer_idx(model, layer_name)
    model.layers[layer_idx].activation = keras.activations.linear
    model = vis.utils.apply_modifications(model)
    img = vis.visualization.visualize_activation(model, layer_idx, filter_indices=filter_index, seed_input=42,
                                                 backprop_modifier=backprop_modifier, grad_modifier=grad_modifier,
                                                 act_max_weight=act_max_weight, lp_norm_weight=lp_norm_weight,
                                                 tv_weight=tv_weight, input_range=input_range, verbose=verbose,
                                                 max_iter=opt_max_iter, input_modifiers=opt_input_mods,
                                                 callbacks=opt_callbacks)
    plt.imshow(img[..., 0])
    fig_name = 'CAM' + str(filter_index) + str(layer_name) + str(model)
    plt.savefig(fig_name)
    return plt

#def CAMList():
