import os
import numpy as np
import concurrent.futures
import load_data
import tensorflow as tf
import matplotlib.pyplot as plt
import ops
import model


# if __name__ == '__main__':
#     img_size = [256, 256]
#
#     img_rgb_path = 'images\\validation\\img_rgb.png'
#     theme_path = 'images\\validation\\theme_rgb.png'
#     theme_mask_path = 'images\\validation\\theme_mask.png'
#     points_path = 'images\\validation\\points_rgb.png'
#     points_mask_path = 'images\\validation\\points_mask.png'
#
#     sess = tf.Session()
#
#     # read image
#     img_rgb = tf.read_file(img_rgb_path)
#     img_rgb = tf.image.decode_png(img_rgb, channels=3)
#     img_rgb = tf.image.resize_images(img_rgb, img_size)
#     img_rgb = tf.cast(img_rgb, tf.float32) / 255.
#     img_rgb = tf.reshape(img_rgb, [1] + img_size + [3])
#
#     theme_rgb = tf.read_file(theme_path)
#     theme_rgb = tf.image.decode_png(theme_rgb, channels=3)
#     theme_rgb = tf.image.resize_images(theme_rgb, [1, 7])
#     theme_rgb = tf.cast(theme_rgb, tf.float32) / 255.
#     theme_rgb = tf.reshape(theme_rgb, [1, 1, 7, 3])
#
#     theme_mask = tf.read_file(theme_mask_path)
#     theme_mask = tf.image.decode_png(theme_mask, channels=1)
#     theme_mask = tf.image.resize_images(theme_mask, [1, 7])
#     theme_mask = tf.cast(theme_mask, tf.float32) / 255.
#     theme_mask = tf.reshape(theme_mask, [1, 1, 7, 1])
#
#     points_rgb = tf.read_file(points_path)
#     points_rgb = tf.image.decode_png(points_rgb, channels=3)
#     points_rgb = tf.image.resize_images(points_rgb, img_size)
#     points_rgb = tf.cast(points_rgb, tf.float32) / 255.
#     points_rgb = tf.reshape(points_rgb, [1] + img_size + [3])
#
#     points_mask = tf.read_file(points_mask_path)
#     points_mask = tf.image.decode_png(points_mask, channels=1)
#     points_mask = tf.image.resize_images(points_mask, img_size)
#     points_mask = tf.cast(points_mask, tf.float32) / 255.
#     points_mask = tf.reshape(points_mask, [1] + img_size + [1])
#
#     # convert color space
#     img_lab = ops.rgb_to_lab(img_rgb)
#     img_l = img_lab[:, :, :, 0] / 100. * 2 - 1
#     img_l = tf.reshape(img_l, [1] + img_size + [1])
#     _, img_l_grad = ops.sobel(img_l)
#
#     theme_lab = ops.rgb_to_lab(theme_rgb)
#     theme_ab = (theme_lab[:, :, :, 1:] + 128) / 255. * 2 - 1
#
#     points_lab = ops.rgb_to_lab(points_rgb)
#     points_ab = (points_lab[:, :, :, 1:] + 128) / 255. * 2 - 1
#
#     # load model
#     output_ab = \
#         model.inference1(img_l, img_l_grad, theme_ab, theme_mask, points_ab, points_mask, False, 'UserGuide')
#     output_lab = tf.concat([(img_l + 1) / 2. * 100, (output_ab + 1) / 2. * 255 - 128], axis=3)
#     output_rgb = ops.lab_to_rgb(output_lab)
#
#     # load check point
#     saver = tf.train.Saver()
#     print('========== Loading Check Point ==========')
#     ckpt = tf.train.get_checkpoint_state('logs/logs_baseline')
#     if ckpt and ckpt.model_checkpoint_path:
#         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print('\033[0;35m\'%s\' load successfully, global step = %s\n\033[0m' % ('UserGuide', global_step))
#     else:
#         print('\033[0;31m\'%s\' load failed.\n\033[0m' % 'UserGuide')
#
#     # show results
#     img_out = sess.run(output_rgb)
#     plt.imshow(img_out[0])
#     plt.show()


if __name__ == '__main__':
    img_size = [256, 256]

    img_rgb_path = 'images\\validation\\img_rgb.png'
    theme_path = 'images\\validation\\theme_rgb.png'
    theme_mask_path = 'images\\validation\\theme_mask.png'
    points_path = 'images\\validation\\points_rgb.png'
    points_mask_path = 'images\\validation\\points_mask.png'
    img_paths = [img_rgb_path, theme_path, theme_mask_path, points_path, points_mask_path]

    mm = model.UserGuide()
    mm.test_one_img(model.inference1, 'UserGuide', [256, 256], img_paths, 'logs/logs_baseline')