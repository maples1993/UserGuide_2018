"""
Date: 2018/9/29
Regular operation
"""
import tensorflow as tf
import tensorflow.contrib.layers as layers


# convolution layer
def conv2d(inputs, output_dim, kernel_size, stride, dilation=1, padding='SAME',
           activation_fn=None, norm_fn=None, is_training=True, scope_name=None):
    """
    Convolution for 2D
    :param inputs: A 4-D tensor
    :param output_dim: A int
    :param kernel_size: A int
    :param stride: A int
    :param dilation: A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn: A function handle
    :param norm_fn: A function handle
    :param is_training: True or False
    :param scope_name: A string
    :return: A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        # convolution
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=output_dim,
                                kernel_size=kernel_size,
                                strides=stride,
                                dilation_rate=dilation,
                                padding=padding,
                                use_bias=False,
                                bias_initializer=layers.xavier_initializer())

        # normalization function
        if norm_fn is None:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            conv = conv + biases
        elif norm_fn is tf.contrib.layers.batch_norm:
            conv = norm_fn(inputs=conv,
                           updates_collections=None,
                           is_training=is_training)
        elif norm_fn is tf.layers.batch_normalization:
            conv = norm_fn(inputs=conv,
                           axis=3,
                           epsilon=1e-5,
                           momentum=0.9,
                           training=is_training,
                           gamma_initializer=tf.random_uniform_initializer(1.0, 0.02))
        elif norm_fn is group_norm:     # Kaiming He《Group Normalization》
            conv = norm_fn(inputs=conv,
                           G=32,
                           eps=1e-5)
        else:
            raise NameError

        # activation function
        if activation_fn is None:
            return conv
        else:
            return activation_fn(conv)


# depth-wise convolution layer(only for user-guided)
def depth_wise_conv2d(inputs, multiplier, kernel_size, stride, padding='SAME',
                      activation_fn=None, scope_name=None):
    """
    Depth-wise convolution for 2D
    :param inputs:  A 4-D tensor
    :param multiplier:  A int
    :param kernel_size: A int
    :param stride:  A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn:   A function handle
    :param scope_name:  A string
    :return:    A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        # convolution
        weights = tf.constant(name='w',
                              value=1.,
                              shape=[kernel_size, kernel_size, inputs.get_shape()[-1], multiplier],
                              dtype=tf.float32)
        conv = tf.nn.depthwise_conv2d(input=inputs,
                                      filter=weights,
                                      strides=[1, stride, stride, 1],
                                      padding=padding)

        # activation function
        if activation_fn is None:
            return conv
        else:
            return activation_fn(conv)


# convolution transpose layer
def conv2d_transpose(inputs, output_dim, kernel_size, stride, padding='SAME',
                     activation_fn=None, norm_fn=None, is_training=True, scope_name=None):
    """
    Deconvolution for 2D
    :param inputs: A 4-D tensor
    :param output_dim: A int
    :param kernel_size: A int
    :param stride: A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn: A function handle
    :param norm_fn: A function handle
    :param is_training: True or False
    :param scope_name: A string
    :return: A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        # deconvolution
        weights = tf.get_variable(name='w',
                                  shape=[kernel_size, kernel_size, output_dim, inputs.get_shape()[-1].value],
                                  dtype=tf.float32,
                                  initializer=layers.xavier_initializer())
        output_shape = inputs.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = output_dim
        deconv = tf.nn.conv2d_transpose(value=inputs,
                                        filter=weights,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)

        # normalization function
        if norm_fn is None:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            deconv = deconv + biases
        elif norm_fn is tf.contrib.layers.batch_norm:
            deconv = norm_fn(inputs=deconv,
                             updates_collections=None,
                             is_training=is_training)
        elif norm_fn is tf.layers.batch_normalization:
            deconv = norm_fn(inputs=deconv,
                             axis=3,
                             epsilon=1e-5,
                             momentum=0.9,
                             training=is_training,
                             gamma_initializer=tf.random_uniform_initializer(1.0, 0.02))
        elif norm_fn is group_norm:  # 何凯明《Group Normalization》
            deconv = norm_fn(inputs=deconv,
                             G=32,
                             eps=1e-5)
        else:
            raise NameError

        # activation function
        if activation_fn is None:
            return deconv
        else:
            return activation_fn(deconv)


# pooling layer
def avg_pool2d(inputs, kernel_size, stride, name=None):
    return tf.nn.avg_pool(inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding='SAME', name=name)


# fully connected layer
def fully_connected(inputs, output_dim, activation_fn=None, norm_fn=None, is_training=True, scope_name=None):
    """
    Fully-connected for 1D
    :param inputs: A 4-D tensor
    :param output_dim: A int
    :param kernel_size: A int
    :param padding: 'SAME' or 'VALID'
    :param activation_fn: A function handle
    :param norm_fn: A function handle
    :param is_training: True or False
    :param scope_name: A string
    :return: A 4-D tensor
    """
    with tf.variable_scope(scope_name):
        weights = tf.get_variable(name='w',
                                  shape=[inputs.get_shape()[-1].value, output_dim],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        fc = tf.matmul(inputs, weights)

        # normalization function
        if norm_fn is None:
            biases = tf.get_variable(name='b',
                                     shape=[output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            fc = fc + biases
        elif norm_fn is tf.contrib.layers.batch_norm:
            fc = norm_fn(inputs=fc,
                         updates_collections=None,
                         is_training=is_training)
        elif norm_fn is tf.layers.batch_normalization:
            fc = norm_fn(inputs=fc,
                         axis=1,
                         epsilon=1e-5,
                         momentum=0.9,
                         training=is_training,
                         gamma_initializer=tf.random_uniform_initializer(1.0, 0.02))
        elif norm_fn is group_norm:  # Kaiming He《Group Normalization》
            fc = norm_fn(inputs=fc,
                         G=32,
                         eps=1e-5)
        else:
            raise NameError

        # activation function
        if activation_fn is None:
            return fc
        else:
            return activation_fn(fc)


# group normalization
def group_norm(inputs, G=32, eps=1e-5, name='GroupNorm'):
    with tf.variable_scope(name):
        N, H, W, C = inputs.shape
        gamma = tf.get_variable(name='gamma',
                                shape=[1, 1, 1, C],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1))
        beta = tf.get_variable(name='beta',
                               shape=[1, 1, 1, C],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())

        inputs = tf.reshape(inputs, [N, G, H, W, C // G])
        mean, var = tf.nn.moments(inputs, [2, 3, 4], keep_dims=True)
        inputs = (inputs - mean) / tf.sqrt(var + eps)
        x = tf.reshape(inputs, [N, H, W, C])

    return x * gamma + beta


# Sobel edge operator
def sobel(image_batch):
    """
    :param image_batch: 4-D or 3-D tensor
    :return: 4-D tensor
    """
    if len(image_batch.shape) == 3:
        image_batch = tf.reshape(image_batch, [image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1])
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
    sobel_y_filter = tf.reshape(sobel_y, [3, 3, 1, 1])
    filtered_x = tf.nn.conv2d(image_batch, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(image_batch, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_xy = tf.concat([filtered_x, filtered_y], axis=3)
    fileterd_merge = tf.sqrt(tf.square(filtered_x) + tf.square(filtered_y) + 1e-5)
    return filtered_xy, fileterd_merge


# convert RGB to LAB
def rgb_to_lab(image_rgb):
    """
    :param image_rgb: 4-D tensor, float32, [0, 1] normalization
    :return: 4-D tensor, float32, no normalization
    """
    assert image_rgb.get_shape()[-1] == 3

    rgb_pixels = tf.reshape(image_rgb, [-1, 3])
    # RGB to XYZ
    with tf.name_scope("rgb_to_xyz"):
        linear_mask = tf.cast(rgb_pixels <= 0.04045, dtype=tf.float32)
        expoential_mask = tf.cast(rgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (rgb_pixels / 12.92) * linear_mask +\
                     (((rgb_pixels + 0.055) / 1.055) ** 2.4) * expoential_mask
        transfer_mat = tf.constant([
            [0.412453, 0.212671, 0.019334],
            [0.357580, 0.715160, 0.119193],
            [0.180423, 0.072169, 0.950227]
        ], dtype=tf.float32)
        xyz_pixels = tf.matmul(rgb_pixels, transfer_mat)

    # XYZ to LAB
    with tf.name_scope("xyz_to_lab"):
        # normalize D65 white point
        xyz_norm_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
        # xyz_norm_pixels = tf.multiply(xyz_pixels, [1 / 0.95047, 1.0, 1 / 1.08883])
        epsilon = 6/29
        linear_mask = tf.cast(xyz_norm_pixels <= epsilon**3, dtype=tf.float32)
        expoential_mask = tf.cast(xyz_norm_pixels > epsilon**3, dtype=tf.float32)
        f_xyf_pixels = (xyz_norm_pixels / (3 * epsilon**2) + 4/29) * linear_mask +\
                       (xyz_norm_pixels**(1/3)) * expoential_mask
        transfer_mat2 = tf.constant([
            [0.0, 500.0, 0.0],
            [116.0, -500.0, 200.0],
            [0.0, 0.0, -200.0]
        ], dtype=tf.float32)
        lab_pixels = tf.matmul(f_xyf_pixels, transfer_mat2) + tf.constant([-16.0, 0.0, 0.0], dtype=tf.float32)

        image_lab = tf.reshape(lab_pixels, tf.shape(image_rgb))

    return image_lab


# convert LAB to RGB
def lab_to_rgb(image_lab):
    """
    :param image_lab: 4-D tensor, float32, no normalization
    :return: 4-D tensor, float32,
    """
    assert image_lab.shape[-1] == 3

    lab_pixels = tf.reshape(image_lab, [-1, 3])
    # LAB to XYZ
    with tf.name_scope('lab_to_xyz'):
        transfer_mat1 = tf.constant([
            [1/116.0, 1/116.0, 1/116.0],
            [1/500.0, 0.0, 0.0],
            [0.0, 0.0, -1/200.0]
        ], dtype=tf.float32)
        fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), transfer_mat1)
        epsilon = 6/29
        linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
        expoential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
        xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask +\
                     (fxfyfz_pixels **3) * expoential_mask
        xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    # XYZ to RGB
    with tf.name_scope('xyz_to_rgb'):
        transfer_mat2 = tf.constant([
            [3.2404542, -0.9692660, 0.0556434],
            [-1.5371385, 1.8760108, -0.2040259],
            [-0.4985314, 0.0415560, 1.0572252]
        ])
        rgb_pixels = tf.matmul(xyz_pixels, transfer_mat2)
        rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
        linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
        expoential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
        rgb_pixels = rgb_pixels * 12.92 * linear_mask +\
                     ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * expoential_mask

        image_rgb = tf.reshape(rgb_pixels, tf.shape(image_lab))

    return image_rgb
