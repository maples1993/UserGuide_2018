import model
import matplotlib.pyplot as plt


def train1():
    color_img_dir = 'D:\\Colorization_Set\\color\\abbey'
    color_map_dir = 'D:\\Colorization_Set\\color_map\\abbey'
    theme_dir = 'D:\\Colorization_Set\\color_theme\\abbey'
    theme_mask_dir = 'D:\\Colorization_Set\\color_theme_mask\\abbey'
    points_dir = 'D:\\Colorization_Set\\local_points\\abbey'
    points_mask_dir = 'D:\\Colorization_Set\\local_points_mask\\abbey'
    img_dirs = [color_img_dir, color_map_dir, theme_dir, theme_mask_dir, points_dir, points_mask_dir]

    train_model = model.UserGuide()
    train_model.load_data(img_dirs=img_dirs,
                          image_size=[256, 256],
                          batch_size=2,
                          is_random=True,
                          check=False)
    train_model.train([256, 256],
                      model.inference1,
                      'UserGuide',
                      'logs_1',
                      300000,
                      1e-3)


if __name__ == '__main__':
    train1()