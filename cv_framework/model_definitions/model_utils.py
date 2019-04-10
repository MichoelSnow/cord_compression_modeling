import gin

@gin.configurable
def set_input_output(input_shape=(None, None, None), output_shape=(None)):
    image_size=(input_shape[0], input_shape[1])
    in_shape = input_shape
    out_shape = output_shape
    return image_size, in_shape, out_shape
