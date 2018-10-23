import keras

def NASNet(input_shape=None, penultimate_filters=4032, num_blocks=6, stem_block_filters=96, skip_reduction=True,
           filter_multiplier=2, pooling=None, classes=3):
    '''Instantiates a NASNet model.'''
    img_input = keras.layers.Input(shape=input_shape)
    if penultimate_filters % (24 * (filter_multiplier ** 2)) != 0:
        raise ValueError(f'For NASNet-A models, the `penultimate_filters` must be a multiple of '
                         f'24 * (`filter_multiplier` ** 2). Current value: {penultimate_filters}')
    filters = penultimate_filters // 24

    x = keras.layers.Conv2D(stem_block_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False,
                            name='stem_conv1', kernel_initializer='he_normal')(img_input)
    x = keras.layers.BatchNormalization(axis=3, momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)
    p = None
    x, p = reduction_a_cell(x, p, filters // (filter_multiplier ** 2), block_id='stem_1')
    x, p = reduction_a_cell(x, p, filters // filter_multiplier, block_id='stem_2')

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id=f'{i}')

    x, p0 = reduction_a_cell(x, p, filters * filter_multiplier, block_id=f'reduce_{num_blocks}')
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier, block_id=f'{num_blocks + i + 1}')

    x, p0 = reduction_a_cell(x, p, filters * filter_multiplier ** 2, block_id=f'reduce_{2 * num_blocks}')
    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = normal_a_cell(x, p, filters * filter_multiplier ** 2, block_id=f'{2 * num_blocks + i + 1}')

    x = keras.layers.Activation('relu')(x)

    if not pooling:
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise ValueError(f'Pooling must be set to None, avg, or max. Current value is {pooling}')

    model = keras.models.Model(img_input, x, name='NASNet')
    return model

def reduction_a_cell(ip, p, filters, block_id=None):
    '''Adds a Reduction cell for NASNet-A (Fig. 4 in the paper).'''

    with keras.backend.name_scope('reduction_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = keras.layers.Activation('relu')(ip)
        h = keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % block_id,
                                use_bias=False, kernel_initializer='he_normal')(h)
        h = keras.layers.BatchNormalization(axis=3, momentum=0.9997, epsilon=1e-3,
                                            name='reduction_bn_1_%s' % block_id)(h)
        h3 = keras.layers.ZeroPadding2D(padding=correct_pad(keras.backend, h, 3), name='reduction_pad_1_%s' % block_id)(h)

        with keras.backend.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), block_id='reduction_left1_%s' % block_id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id='reduction_right1_%s' % block_id)
            x1 = keras.layers.add([x1_1, x1_2], name='reduction_add_1_%s' % block_id)

        with keras.backend.name_scope('block_2'):
            x2_1 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid',
                                             name='reduction_left2_%s' % block_id)(h3)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id='reduction_right2_%s' % block_id)
            x2 = keras.layers.add([x2_1, x2_2], name='reduction_add_2_%s' % block_id)

        with keras.backend.name_scope('block_3'):
            x3_1 = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='valid',
                                                 name='reduction_left3_%s' % block_id)(h3)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), block_id='reduction_right3_%s' % block_id)
            x3 = keras.layers.add([x3_1, x3_2], name='reduction_add3_%s' % block_id)

        with keras.backend.name_scope('block_4'):
            x4 = keras.layers.AveragePooling2D((3, 3),strides=(1, 1), padding='same',
                                               name='reduction_left4_%s' % block_id)(x1)
            x4 = keras.layers.add([x2, x4])

        with keras.backend.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), block_id='reduction_left4_%s' % block_id)
            x5_2 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid',
                                             name='reduction_right5_%s' % block_id)(h3)
            x5 = keras.layers.add([x5_1, x5_2], name='reduction_add4_%s' % block_id)

        x = keras.layers.concatenate(
            [x2, x3, x4, x5],
            axis=3,
            name='reduction_concat_%s' % block_id)
        return x, ip

def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), block_id=None):
    with keras.backend.name_scope('separable_conv_block_%s' % block_id):
        x = keras.layers.Activation('relu')(ip)
        if strides == (2, 2):
            x = keras.layers.ZeroPadding2D(
                padding=correct_pad(x, kernel_size),
                name='separable_conv_1_pad_%s' % block_id)(x)
            conv_pad = 'valid'
        else:
            conv_pad = 'same'
        x = keras.layers.SeparableConv2D(filters, kernel_size,
                                   strides=strides,
                                   name='separable_conv_1_%s' % block_id,
                                   padding=conv_pad, use_bias=False,
                                   kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization(
            momentum=0.9997,
            epsilon=1e-3,
            name='separable_conv_1_bn_%s' % (block_id))(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(filters, kernel_size,
                                   name='separable_conv_2_%s' % block_id,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization(
            momentum=0.9997,
            epsilon=1e-3,
            name='separable_conv_2_bn_%s' % (block_id))(x)
    return x

def _adjust_block(p, ip, filters, block_id=None):
    channel_dim = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    img_dim = 2 if keras.backend.image_data_format() == 'channels_first' else -2
    ip_shape = keras.backend.int_shape(ip)

    if p is not None:
        p_shape = keras.backend.int_shape(p)

    with keras.backend.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p_shape[img_dim] != ip_shape[img_dim]:
            with keras.backend.name_scope('adjust_reduction_block_%s' % block_id):
                p = keras.layers.Activation('relu',
                                      name='adjust_relu_1_%s' % block_id)(p)
                p1 = keras.layers.AveragePooling2D(
                    (1, 1),
                    strides=(2, 2),
                    padding='valid',
                    name='adjust_avg_pool_1_%s' % block_id)(p)
                p1 = keras.layers.Conv2D(
                    filters // 2, (1, 1),
                    padding='same',
                    use_bias=False, name='adjust_conv_1_%s' % block_id,
                    kernel_initializer='he_normal')(p1)

                p2 = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = keras.layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = keras.layers.AveragePooling2D(
                    (1, 1),
                    strides=(2, 2),
                    padding='valid',
                    name='adjust_avg_pool_2_%s' % block_id)(p2)
                p2 = keras.layers.Conv2D(
                    filters // 2, (1, 1),
                    padding='same',
                    use_bias=False,
                    name='adjust_conv_2_%s' % block_id,
                    kernel_initializer='he_normal')(p2)

                p = keras.layers.concatenate([p1, p2])
                p = keras.layers.BatchNormalization(
                    momentum=0.9997,
                    epsilon=1e-3,
                    name='adjust_bn_%s' % block_id)(p)

        elif p_shape[channel_dim] != filters:
            with keras.backend.name_scope('adjust_projection_block_%s' % block_id):
                p = keras.layers.Activation('relu')(p)
                p = keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                                        name=f'adjust_conv_projection_{block_id}', use_bias=False,
                                        kernel_initializer='he_normal')(p)
                p = keras.layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name=f'adjust_bn_{block_id}')(p)
    return p

def _normal_a_cell(ip, p, filters, block_id=None):
    '''Adds a Normal cell for NASNet-A (Fig. 4 in the paper).'''
    with keras.backend.name_scope('normal_A_block_%s' % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = keras.layers.Activation('relu')(ip)
        h = keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % block_id,
                                use_bias=False, kernel_initializer='he_normal')(h)
        h = keras.layers.BatchNormalization(axis=3, momentum=0.9997, epsilon=1e-3, name='normal_bn_1_%s' % block_id)(h)

        with keras.backend.name_scope('block_1'):
            x1_1 = _separable_conv_block( h, filters, kernel_size=(5, 5), block_id='normal_left1_%s' % block_id)
            x1_2 = _separable_conv_block( p, filters, block_id='normal_right1_%s' % block_id)
            x1 = keras.layers.add([x1_1, x1_2], name='normal_add_1_%s' % block_id)

        with keras.backend.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), block_id='normal_left2_%s' % block_id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), block_id='normal_right2_%s' % block_id)
            x2 = keras.layers.add([x2_1, x2_2], name='normal_add_2_%s' % block_id)

        with keras.backend.name_scope('block_3'):
            x3 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                               name='normal_left3_%s' % (block_id))(h)
            x3 = keras.layers.add([x3, p], name='normal_add_3_%s' % block_id)

        with keras.backend.name_scope('block_4'):
            x4_1 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                 name='normal_left4_%s' % (block_id))(p)
            x4_2 = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                 name='normal_right4_%s' % (block_id))(p)
            x4 = keras.layers.add([x4_1, x4_2], name='normal_add_4_%s' % block_id)

        with keras.backend.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, block_id='normal_left5_%s' % block_id)
            x5 = keras.layers.add([x5, h], name='normal_add_5_%s' % block_id)

        x = keras.layers.concatenate([p, x1, x2, x3, x4, x5], axis=3, name='normal_concat_%s' % block_id)
    return x, ip



def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        kernel_size: An integer or tuple/list of 2 integers.
    """
    input_size = keras.backend.int_shape(inputs)[1:3]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))