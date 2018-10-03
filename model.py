"""
Date: 2018/9/29
Model construction
"""
import tensorflow as tf
import numpy as np
import ops
import load_data
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import concurrent.futures


# Baseline 2018/9/30
def inference1(img_l_batch, img_l_gra_batch, theme_ab_batch, theme_mask_batch, local_ab_batch, local_mask_batch,
               is_training=True, scope_name='UserGuide'):
    """
    :param img_l_batch: l channel of input image
    :param img_l_gra_batch: sobel edge map of l channel of input image
    :param theme_ab_batch: ab channel of input color theme
    :param theme_mask_batch: theme mask
    :param local_ab_batch: ab channel of input local points
    :param local_mask_batch: local points mask
    :param is_training: bool, indicate usage of model (training or testing)
    :param scope_name: model name
    :return: ab channel of output image
    """
    assert img_l_batch.get_shape()[-1] == 1
    assert img_l_gra_batch.get_shape()[-1] == 1     # horizontal and vertical direction
    assert theme_ab_batch.get_shape()[-1] == 2
    assert theme_mask_batch.get_shape()[-1] == 1
    assert local_ab_batch.get_shape()[-1] == 2
    assert local_mask_batch.get_shape()[-1] == 1

    ngf = 64
    theme_batch = tf.concat([theme_ab_batch, theme_mask_batch], axis=3)
    local_batch = tf.concat([local_ab_batch, local_mask_batch], axis=3)
    print('Image  Inputs:', img_l_batch)
    print('Theme  Inputs:', theme_batch)
    print('Points Inputs:', local_batch)
    print()

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        theme_batch = tf.reshape(theme_batch, [img_l_batch.get_shape()[0], 1, 1, -1])
        glob_conv1 = ops.conv2d(theme_batch, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                                norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv1')
        glob_conv2 = ops.conv2d(glob_conv1, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                                norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv2')
        glob_conv3 = ops.conv2d(glob_conv2, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                                norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv3')
        glob_conv4 = ops.conv2d(glob_conv3, ngf * 8, 1, 1, activation_fn=tf.nn.relu,
                                norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='glob_conv4')
        print('ThemeBlock', glob_conv4)

        ab_conv1_1 = ops.conv2d(local_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                                norm_fn=None, is_training=is_training, scope_name='ab_conv1_1')
        bw_conv1_1 = ops.conv2d(img_l_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                                norm_fn=None, is_training=is_training, scope_name='bw_conv1_1')
        gra_conv1_1 = ops.conv2d(img_l_gra_batch, ngf, 3, 1, activation_fn=tf.nn.relu,
                                 norm_fn=None, is_training=is_training, scope_name='gra_conv1_1')
        print('LocalBlock', gra_conv1_1)

        conv1_1 = ab_conv1_1 + bw_conv1_1 + gra_conv1_1  # TODO: Merge Local Points and Gradient Maps
        conv1_1 = ops.conv2d(conv1_1, ngf, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv1_1')
        conv1_2 = ops.conv2d(conv1_1, ngf, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv1_2')
        conv1_2_ss = ops.depth_wise_conv2d(conv1_2, 1, 1, 2, activation_fn=None, scope_name='conv1_2_ss')
        print('ConvBlock 1', conv1_2_ss)

        conv2_1 = ops.conv2d(conv1_2_ss, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv2_1')
        conv2_2 = ops.conv2d(conv2_1, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv2_2')
        conv2_2_ss = ops.depth_wise_conv2d(conv2_2, 1, 1, 2, activation_fn=None, scope_name='conv2_2_ss')
        print('ConvBlock 2', conv2_2_ss)

        conv3_1 = ops.conv2d(conv2_2_ss, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv3_1')
        conv3_2 = ops.conv2d(conv3_1, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv3_2')
        conv3_3 = ops.conv2d(conv3_2, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv3_3')
        conv3_3_ss = ops.depth_wise_conv2d(conv3_3, 1, 1, 2, activation_fn=None, scope_name='conv3_3_ss')
        print('ConvBlock 3', conv3_3_ss)

        conv4_1 = ops.conv2d(conv3_3_ss, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv4_1')
        conv4_2 = ops.conv2d(conv4_1, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv4_2')
        conv4_3 = ops.conv2d(conv4_2, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv4_3')
        print('ConvBlock 4', conv4_3)

        conv4_3 = conv4_3 + glob_conv4      # TODO: Merge Color Theme
        conv5_1 = ops.conv2d(conv4_3, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv5_1')
        conv5_2 = ops.conv2d(conv5_1, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv5_2')
        conv5_3 = ops.conv2d(conv5_2, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv5_3')
        print('ConvBlock 5', conv5_3)

        conv6_1 = ops.conv2d(conv5_3, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv6_1')
        conv6_2 = ops.conv2d(conv6_1, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv6_2')
        conv6_3 = ops.conv2d(conv6_2, ngf * 8, 3, 1, dilation=2, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv6_3')
        print('ConvBlock 6', conv6_3)

        conv7_1 = ops.conv2d(conv6_3, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv7_1')
        conv7_2 = ops.conv2d(conv7_1, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv7_2')
        conv7_3 = ops.conv2d(conv7_2, ngf * 8, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv7_3')
        print('ConvBlock 7', conv7_3)

        conv3_3_short = ops.conv2d(conv3_3, ngf * 4, 3, 1, activation_fn=None,
                               is_training=is_training, scope_name='conv3_3_short')
        conv8_1 = ops.conv2d_transpose(conv7_3, ngf * 4, 4, 2, activation_fn=None,
                                   is_training=is_training, scope_name='conv8_1')
        conv8_1_comb = tf.nn.relu(conv3_3_short + conv8_1)
        conv8_2 = ops.conv2d(conv8_1_comb, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=None, is_training=is_training, scope_name='conv8_2')
        conv8_3 = ops.conv2d(conv8_2, ngf * 4, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv8_3')
        print('ConvBlock 8', conv8_3)

        conv2_2_short = ops.conv2d(conv2_2, ngf * 2, 3, 1, activation_fn=None,
                                   is_training=is_training, scope_name='conv2_2_short')
        conv9_1 = ops.conv2d_transpose(conv8_3, ngf * 2, 4, 2, activation_fn=None,
                                       is_training=is_training, scope_name='conv9_1')
        conv9_1_comb = tf.nn.relu(conv2_2_short + conv9_1)
        conv9_2 = ops.conv2d(conv9_1_comb, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                             norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv9_2')
        print('ConvBlock 9', conv9_2)

        conv1_2_short = ops.conv2d(conv1_2, ngf * 2, 3, 1, activation_fn=None,
                                   is_training=is_training, scope_name='conv1_2_short')
        conv10_1 = ops.conv2d_transpose(conv9_2, ngf * 2, 4, 2, activation_fn=None,
                                        is_training=is_training, scope_name='conv10_1')
        conv10_1_comb = tf.nn.relu(conv1_2_short + conv10_1)
        conv10_2 = ops.conv2d(conv10_1_comb, ngf * 2, 3, 1, activation_fn=tf.nn.relu,
                              norm_fn=tf.layers.batch_normalization, is_training=is_training, scope_name='conv10_2')
        print('ConvBlock 10', conv10_2)

        conv10_ab = ops.conv2d(conv10_2, 2, 1, 1, activation_fn=tf.nn.tanh,
                               norm_fn=None, is_training=is_training, scope_name='conv10_ab')
        print('OutputBlock', conv10_ab, end='\n\n')

    return conv10_ab


# model class
class UserGuide:
    def __init__(self):
        # input file list
        self.input_file_list = dict()
        self.input_file_list['img_rgb'] = np.ndarray([0])
        self.input_file_list['color_map'] = np.ndarray([0])
        self.input_file_list['theme'] = np.ndarray([0])
        self.input_file_list['theme_mask'] = np.ndarray([0])
        self.input_file_list['points'] = np.ndarray([0])
        self.input_file_list['points_mask'] = np.ndarray([0])
        # model related
        self.model_input = dict()
        self.model_input['img_rgb'] = None      # original image
        self.model_input['img_lab'] = None
        self.model_input['img_l'] = 0.
        self.model_input['img_l_grad'] = None
        self.model_input['img_ab'] = None
        self.model_input['img_ab_grad'] = None
        self.model_input['color_map_rgb'] = None    # color map
        self.model_input['color_map_lab'] = None
        self.model_input['color_map_ab'] = None
        self.model_input['theme_rgb'] = None        # global color theme
        self.model_input['theme_lab'] = None
        self.model_input['theme_ab'] = None
        self.model_input['theme_mask'] = None
        self.model_input['points_rgb'] = None       # local points
        self.model_input['points_lab'] = None
        self.model_input['points_ab'] = None
        self.model_input['points_mask'] = None
        self.model_input['output_ab'] = None        # output of model
        self.model_input['output_ab_grad'] = None
        self.model_input['output_rgb'] = None
        # test image
        self.img_test = dict()
        self.img_test['img_rgb'] = None
        self.img_test['img_lab'] = None
        self.img_test['img_l'] = None
        self.img_test['img_l_grad'] = None
        self.img_test['img_ab'] = None
        self.img_test['img_ab_grad'] = None
        self.img_test['theme_rgb'] = None
        self.img_test['theme_lab'] = None
        self.img_test['theme_ab'] = None
        self.img_test['theme_mask'] = None
        self.img_test['points_rgb'] = None
        self.img_test['points_lab'] = None
        self.img_test['points_ab'] = None
        self.img_test['points_mask'] = None
        self.img_test['output_ab'] = None
        self.img_test['output_ab_grad'] = None
        self.img_test['output_rgb'] = None
        self.img_test['psnr'] = 0.
        # Tensorflow
        self.sess = tf.Session()
        self.model_vars = dict()       # trainable parameters of model
        self.savers = dict()

    # load file list from directory
    def __load_file_list(self, file_dir, file_type):
        """
        :param file_dir: str
        :param file_type: str, file type of file
        :return: None
        """
        file_list = load_data.get_all_files(file_dir)
        if file_type in self.input_file_list:
            self.input_file_list[file_type] = file_list
            print('\033[0;35m%-70s\tload successfully, item nums: %8d.\033[0m'
                  % (file_dir+'('+file_type+')', len(self.input_file_list[file_type])))
        else:
            raise ValueError('Wrong file type: load_file_list(%s, %s)' % (file_dir, file_type))

    # shuffle input file list
    def __shuffle_file_list(self):
        if len(self.input_file_list['img_rgb']) == len(self.input_file_list['color_map']) == \
                len(self.input_file_list['theme']) == len(self.input_file_list['theme_mask']) == \
                len(self.input_file_list['points']) == len(self.input_file_list['points_mask']) > 0:
            rnd_index = np.arange(len(self.input_file_list['img_rgb']))
            np.random.shuffle(rnd_index)
            for key in self.input_file_list:
                self.input_file_list[key] = self.input_file_list[key][rnd_index]
        else:
            raise AssertionError('length of input list is zero or not consistent:\n'
                                 'img_rgb_list: %8d\n'
                                 'color_map_list: %8d'
                                 'theme_list: %8d\n'
                                 'theme_mask_list: %8d\n'
                                 'points_list: %8d\n'
                                 'points_mask_list: %8d\n' %
                                 (len(self.input_file_list['img_rgb']),
                                  len(self.input_file_list['color_map']),
                                  len(self.input_file_list['theme']),
                                  len(self.input_file_list['theme_mask']),
                                  len(self.input_file_list['points']),
                                  len(self.input_file_list['points_mask'])))

    # load image from file path queue or given path
    def __load_image(self, image_size, ext, channels, file_type, img_path=None):
        """
        If img_path is None, then read an image from the queue, otherwise read an image from the given path.
        :param image_size: 2-D list[int]
        :param ext: str, image extension
        :param channels: channels of input image
        :param file_type: str, file_type of file
        :param img_path: single image path
        :return: 3-D image tensor
        """
        if not img_path:
            filepath_queue = tf.train.slice_input_producer([self.input_file_list[file_type]], shuffle=False)
            img = tf.read_file(filepath_queue[0])
        else:
            img = tf.read_file(img_path)

        if ext == 'jpg':
            img = tf.image.decode_jpeg(img, channels=channels)
        elif ext == 'png':
            img = tf.image.decode_jpeg(img, channels=channels)
        elif ext == 'bmp':
            img = tf.image.decode_bmp(img, channels=channels)
        else:
            raise ValueError('Unknown image type: %s' % ext)
        img = tf.cast(img, tf.float32) / 255.
        img = tf.image.resize_images(img, image_size)

        if not img_path:
            img = tf.reshape(img, image_size + [channels])
        else:
            img = tf.reshape(img, [1] + image_size + [channels])

        return img

    # load images for validation
    def __load_validate_input(self, image_size, img_paths):
        """
        :param image_size: 2-D list[int]
        :param img_paths: list[str]
        :return: None
        """
        img_rgb_path, theme_path, theme_mask_path, points_path, points_mask_path = img_paths
        self.img_test['img_rgb'] = self.__load_image(image_size, 'png', 3, 'img_rgb', img_rgb_path)
        self.img_test['img_lab'] = ops.rgb_to_lab(self.img_test['img_rgb'])
        self.img_test['img_l'] = \
            tf.reshape(self.img_test['img_lab'][:, :, :, 0] / 100. * 2 - 1, [1] + image_size + [1])
        _, self.img_test['img_l_grad'] = ops.sobel(self.img_test['img_l'])
        self.img_test['img_ab'] = (self.img_test['img_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1
        _, self.img_test['img_ab_grad'] = ops.sobel(tf.concat([self.img_test['img_ab'][:, :, :, 0],
                                                              self.img_test['img_ab'][:, :, :, 1]],
                                                              axis=0))

        self.img_test['theme_rgb'] = self.__load_image([1, 7], 'png', 3, 'theme_rgb', theme_path)
        self.img_test['theme_lab'] = ops.rgb_to_lab(self.img_test['theme_rgb'])
        self.img_test['theme_ab'] = (self.img_test['theme_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1
        self.img_test['theme_mask'] = self.__load_image([1, 7], 'png', 1, 'theme_mask', theme_mask_path)

        self.img_test['points_rgb'] = self.__load_image(image_size, 'png', 3, 'points_rgb', points_path)
        self.img_test['points_lab'] = ops.rgb_to_lab(self.img_test['points_rgb'])
        self.img_test['points_ab'] = (self.img_test['points_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1
        self.img_test['points_mask'] = self.__load_image(image_size, 'png', 1, 'points_mask', points_mask_path)

    # load batch of image tensor
    def __load_input_batch(self, batch_size, image_size, is_random=True, blank_rate=None):
        """
        :param batch_size: int
        :param image_size: 2-D list[int]
        :param is_random: bool
        :param blank_rate: 2-D list[double], rate of blank input
        :return: None
        """
        if len(self.input_file_list['img_rgb']) == len(self.input_file_list['color_map']) == \
                len(self.input_file_list['theme']) == len(self.input_file_list['theme_mask']) == \
                len(self.input_file_list['points']) == len(self.input_file_list['points_mask']) > 0:

            print('%8d samples in total.\n' % len(self.input_file_list['img_rgb']))

            img_rgb = self.__load_image(image_size, 'png', 3, 'img_rgb')
            color_map = self.__load_image(image_size, 'png', 3, 'color_map')
            theme = self.__load_image([1, 7], 'png', 3, 'theme')
            theme_mask = self.__load_image([1, 7], 'png', 1, 'theme_mask')
            points = self.__load_image(image_size, 'png', 3, 'points')
            points_mask = self.__load_image(image_size, 'png', 1, 'points_mask')

            # blank input
            color_map_blank = img_rgb
            theme_blank = tf.zeros([1, 7, 3], dtype=tf.float32)
            theme_mask_blank = tf.zeros([1, 7, 1], dtype=tf.float32)
            points_blank = tf.zeros(image_size + [3], dtype=tf.float32)
            points_mask_blank = tf.zeros(image_size + [1], dtype=tf.float32)

            def f1():   # only color theme
                return color_map, theme, theme_mask, points_blank, points_mask_blank

            def f2():   # only local points
                return color_map_blank, theme_blank, theme_mask_blank, points, points_mask

            def f3():   # color theme & local points
                return color_map, theme, theme_mask, points, points_mask

            rnd = tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32)
            rnd = rnd[0]
            if is_random and blank_rate:
                flag1 = tf.less(rnd, blank_rate[0])
                flag2 = tf.logical_and(tf.greater_equal(rnd, blank_rate[0]), tf.less(rnd, sum(blank_rate)))
                flag3 = tf.greater_equal(rnd, sum(blank_rate))
                color_map, theme, theme_mask, points, points_mask = \
                    tf.case({flag1: f1, flag2: f2, flag3: f3}, exclusive=True)

            # original input
            if is_random:
                self.model_input['img_rgb'], self.model_input['color_map_rgb'], \
                self.model_input['theme_rgb'], self.model_input['theme_mask'], \
                self.model_input['points_rgb'], self.model_input['points_mask'] = \
                    tf.train.shuffle_batch(tensors=[img_rgb, color_map, theme, theme_mask, points, points_mask],
                                           batch_size=batch_size,
                                           capacity=1000,
                                           min_after_dequeue=500,
                                           num_threads=4)
            else:
                self.model_input['img_rgb'], self.model_input['color_map_rgb'], \
                self.model_input['theme_rgb'], self.model_input['theme_mask'], \
                self.model_input['points_rgb'], self.model_input['points_mask'] = \
                    tf.train.batch(tensors=[img_rgb, color_map, theme, theme_mask, points, points_mask],
                                   batch_size=1,
                                   capacity=500,
                                   num_threads=1)

            # convert to lab color space
            self.model_input['img_lab'] = ops.rgb_to_lab(self.model_input['img_rgb'])
            self.model_input['img_l'] = \
                tf.reshape(self.model_input['img_lab'][:, :, :, 0] / 100. * 2 - 1, [batch_size] + image_size + [1])
            _, self.model_input['img_l_grad'] = ops.sobel(self.model_input['img_l'])
            self.model_input['img_ab'] = (self.model_input['img_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1
            _, self.model_input['img_ab_grad'] = ops.sobel(tf.concat([self.model_input['img_ab'][:, :, :, 0],
                                                                     self.model_input['img_ab'][:, :, :, 1]],
                                                                     axis=0))

            self.model_input['color_map_lab'] = ops.rgb_to_lab(self.model_input['color_map_rgb'])
            self.model_input['color_map_ab'] = (self.model_input['color_map_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1

            self.model_input['theme_lab'] = ops.rgb_to_lab(self.model_input['theme_rgb'])
            self.model_input['theme_ab'] = (self.model_input['theme_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1

            self.model_input['points_lab'] = ops.rgb_to_lab(self.model_input['points_rgb'])
            self.model_input['points_ab'] = (self.model_input['points_lab'][:, :, :, 1:] + 128.) / 255. * 2 - 1

        else:
            raise AssertionError('length of input list is zero or not consistent:\n'
                                 'img_rgb_list: %8d\n'
                                 'color_map_list: %8d\n'
                                 'theme_list: %8d\n'
                                 'theme_mask_list: %8d\n'
                                 'points_list: %8d\n'
                                 'points_mask_list: %8d\n' %
                                 (len(self.input_file_list['img_rgb']),
                                  len(self.input_file_list['color_map']),
                                  len(self.input_file_list['theme']),
                                  len(self.input_file_list['theme_mask']),
                                  len(self.input_file_list['points']),
                                  len(self.input_file_list['points_mask'])))

    # build model
    def __build_model(self, model, model_input, is_training, scope_name):
        """
        :param model: function handle
        :param model_input: dict
        :param is_training: bool
        :param scope_name: str
        :return: None
        """
        print('========== Building Model ==========')
        model_input['output_ab'] = model(img_l_batch=model_input['img_l'],
                                         img_l_gra_batch=model_input['img_l_grad'],
                                         theme_ab_batch=model_input['theme_ab'],
                                         theme_mask_batch=model_input['theme_mask'],
                                         local_ab_batch=model_input['points_ab'],
                                         local_mask_batch=model_input['points_mask'],
                                         is_training=is_training,
                                         scope_name=scope_name)
        _, model_input['output_ab_grad'] = ops.sobel(tf.concat([model_input['output_ab'][:, :, :, 0],
                                                               model_input['output_ab'][:, :, :, 1]],
                                                               axis=0))
        model_input['output_rgb'] = ops.lab_to_rgb(tf.concat([(model_input['img_l'] + 1.) / 2 * 100,
                                                              (model_input['output_ab'] + 1.) / 2 * 255 - 128],
                                                             axis=3))
        self.model_vars[scope_name] = [var for var in tf.global_variables() if var.name.startswith(scope_name)]
        self.savers[scope_name] = tf.train.Saver(var_list=self.model_vars[scope_name])
        paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in self.model_vars[scope_name]])
        print('\033[0;35mModel \'%s\' load successfully, parameters in total: %8d\n \033[0m' %
              (scope_name, self.sess.run(paras_count)))

    # loss1 (color map + gradient + local points)
    def __loss1(self, alpha1=0.9, alpha2=0.1, alpha3=10):
        """
        :param alpha1: original img
        :param alpha2: color map
        :param alpha3: gradient map
        :param alpha4: local points
        :return:
        """
        loss1 = tf.losses.huber_loss(self.model_input['output_ab'], self.model_input['img_ab']) * alpha1
        loss2 = tf.losses.huber_loss(self.model_input['output_ab'], self.model_input['color_map_ab']) * alpha2
        loss3 = tf.reduce_mean(tf.square(self.model_input['output_ab_grad'] - self.model_input['img_ab_grad'])) * alpha3

        loss_total = loss1 + loss2 + loss3

        return loss_total, [loss1, loss2, loss3]

    # loss2 (mse)
    def __loss2(self, predict, ground):
        """
        :param predict: output of model
        :param ground: ground truth
        :return: double
        """
        return tf.reduce_mean(tf.square(predict - ground))

    # load check_point
    def __load_check_point(self, check_point_path, scope_name):
        """
        :param check_points_path: str
        :param scope_name: str
        :return: None
        """
        print('========== Loading Check Point ==========')
        ckpt = tf.train.get_checkpoint_state(check_point_path)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            self.savers[scope_name].restore(self.sess, ckpt.model_checkpoint_path)
            print('\033[0;35m\'%s\' load successfully, global step = %s\n\033[0m' % (scope_name, global_step))
        else:
            print('\033[0;31m\'%s\' load failed.\n\033[0m' % scope_name)

    # test single image (validate the training process)
    def __test_single_img(self, model, scope_name, image_size, img_paths):
        """
        :param model:
        :param scope_name:
        :param image_size:
        :param img_paths:
        :return:
        """
        self.__load_validate_input(image_size, img_paths)
        self.__build_model(model, self.img_test, False, scope_name)
        self.img_test['psnr'] = tf.image.psnr(self.img_test['img_rgb'], self.img_test['output_rgb'], 1)[0]

    # load data
    def load_data(self, img_dirs, image_size, batch_size, is_random=True, check=False):
        """
        :param img_dirs: list[str], collection of input directory
        :param image_size: 2-D list[int]
        :param batch_size: int
        :param is_random: bool
        :param check: bool, check if the input is valid
        :return: None
        """
        color_img_dir, color_map_dir, theme_dir, theme_mask_dir, points_dir, points_mask_dir = img_dirs
        print('\n========== Loading Data ==========')
        self.__load_file_list(color_img_dir, 'img_rgb')
        self.__load_file_list(color_map_dir, 'color_map')
        self.__load_file_list(theme_dir, 'theme')
        self.__load_file_list(theme_mask_dir, 'theme_mask')
        self.__load_file_list(points_dir, 'points')
        self.__load_file_list(points_mask_dir, 'points_mask')
        if is_random:
            self.__shuffle_file_list()
        self.__load_input_batch(batch_size, image_size, is_random, [0.05, 0.05])

        # check input
        if check:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

            try:
                for step in range(5):
                    if coord.should_stop():
                        break
                    img_l, img_l_grad, img_rgb, color_map, theme, theme_mask, points, points_mask, img_ab, img_ab_grad = \
                        self.sess.run([self.model_input['img_l'], self.model_input['img_l_grad'],
                                       self.model_input['img_rgb'], self.model_input['color_map_rgb'],
                                       self.model_input['theme_rgb'], self.model_input['theme_mask'],
                                       self.model_input['points_rgb'], self.model_input['points_mask'],
                                       self.model_input['img_ab'], self.model_input['img_ab_grad']])
                    img_l = (img_l + 1.) / 2
                    # img_l_grad = (img_l_grad + 1.) / 2
                    img_ab = (img_ab + 1.) / 2
                    # img_ab_grad = (img_ab_grad + 1.) / 2
                    plt.subplot(2, 7, 1), plt.imshow(img_l[0, :, :, 0], 'gray'), plt.title('L channel')
                    plt.subplot(2, 7, 8), plt.imshow(img_rgb[0]), plt.title('ground truth')
                    plt.subplot(2, 7, 2), plt.imshow(theme[0]), plt.title('theme')
                    plt.subplot(2, 7, 9), plt.imshow(theme_mask[0, :, :, 0], 'gray'), plt.title('theme mask')
                    plt.subplot(2, 7, 3), plt.imshow(points[0]), plt.title('points')
                    plt.subplot(2, 7, 10), plt.imshow(points_mask[0, :, :, 0], 'gray'), plt.title('points mask')
                    plt.subplot(2, 7, 4), plt.imshow(img_l_grad[0, :, :, 0], 'gray'), plt.title('L x-sobel')
                    plt.subplot(2, 7, 11), plt.imshow(img_l_grad[0, :, :, 1], 'gray'), plt.title('L y-sobel')
                    plt.subplot(2, 7, 5), plt.imshow(img_ab[0, :, :, 0], 'gray'), plt.title('a channel')
                    plt.subplot(2, 7, 12), plt.imshow(img_ab[0, :, :, 1], 'gray'), plt.title('b channel')
                    plt.subplot(2, 7, 6), plt.imshow(img_ab_grad[0, :, :, 0], 'gray'), plt.title('a channel x-sobel')
                    plt.subplot(2, 7, 13), plt.imshow(img_ab_grad[batch_size, :, :, 0], 'gray'), plt.title('b channel x-sobel')
                    plt.subplot(2, 7, 7), plt.imshow(img_ab_grad[0, :, :, 1], 'gray'), plt.title('a channel y-sobel')
                    plt.subplot(2, 7, 14), plt.imshow(img_ab_grad[batch_size, :, :, 1], 'gray'), plt.title('b channel y-sobel')
                    plt.show()
            except tf.errors.OutOfRangeError:
                print('Done')
            finally:
                coord.request_stop()

            coord.join(threads=threads)
            self.sess.close()

    # train model
    def train(self, image_size, model, scope_name, logs_dir, max_step, lr_init):
        # build model
        self.__build_model(model, self.model_input, True, scope_name)

        # validate model
        img_rgb_path = 'images\\validation\\img_rgb.png'
        theme_path = 'images\\validation\\theme_rgb.png'
        theme_mask_path = 'images\\validation\\theme_mask.png'
        points_path = 'images\\validation\\points_rgb.png'
        points_mask_path = 'images\\validation\\points_mask.png'
        img_paths = [img_rgb_path, theme_path, theme_mask_path, points_path, points_mask_path]
        self.__test_single_img(model, scope_name, image_size, img_paths)

        # preparation for training
        global_step = tf.train.get_or_create_global_step(self.sess.graph)
        train_loss, etc_loss = self.__loss1(0.9, 0.1, 10)
        lr = tf.train.exponential_decay(lr_init, global_step, 1e4, 0.7, staircase=True)
        train_op = tf.train.AdamOptimizer(lr).\
            minimize(train_loss, var_list=self.model_vars[scope_name], global_step=global_step)

        self.sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        s_t = time.time()
        try:
            for step in range(max_step+1):
                if coord.should_stop():
                    break

                _, loss, loss_etc, learning_rate = self.sess.run([train_op, train_loss, etc_loss, lr])

                if step % 100 == 0:
                    runtime = time.time() - s_t
                    psnr = self.sess.run(self.img_test['psnr'])
                    if not etc_loss:
                        print('Step: %6d, loss: %.6f, psnr: %.2fdB, lr: %g, time: %.2fs, time left: %.2fhours' %
                              (step, loss, psnr, learning_rate, runtime, (max_step - step) * runtime / 360000))
                    else:
                        print('Step: %6d, l_total: %.6f, l_ori: %.6f, l_color: %.6f, l_grad: %.6f,'
                              ' psnr: %.2fdB, lr: %g, time: %.2fs, time left: %.2fhours' %
                              (step, loss, loss_etc[0], loss_etc[1], loss_etc[2],
                               psnr, learning_rate, runtime, (max_step - step) * runtime / 360000))
                    s_t = time.time()

                if step % 1000 == 0:
                    img, img_gt, img_val = \
                        self.sess.run([self.img_test['output_rgb'], self.img_test['img_rgb'], self.img_test['psnr']])
                    save_path = 'images\\logs_output\\' + logs_dir
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    img = np.concatenate([img, img_gt], axis=2)
                    img = img[0]
                    plt.imsave(save_path + '\\step_%d_%.2fdB.bmp' % (step, img_val), img)
                    # img = Image.fromarray(img * 255., mode='RGB')
                    # img.convert('RGB').save(save_path + '\\step_{0}_{1}dB.bmp'.format(step, img_val))

                if step % 5000 == 0:
                    checkpoint_path = os.path.join('logs\\' + logs_dir, 'model.ckpt')
                    self.savers[scope_name].save(self.sess, checkpoint_path, global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Done')
        finally:
            coord.request_stop()

        coord.join(threads=threads)
        self.sess.close()

    # test one image
    def test_one_img(self, model, scope_name, image_size, img_paths, logs_dir):
        self.__test_single_img(model, scope_name, image_size, img_paths)
        self.__load_check_point(logs_dir, scope_name)
        img_out, psnr = self.sess.run([self.img_test['output_rgb'], self.img_test['psnr']])
        plt.imshow(img_out[0])
        plt.title(str(psnr))
        plt.show()

