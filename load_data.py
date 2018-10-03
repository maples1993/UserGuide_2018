"""
Date: 2018/9/29
File regular operation
"""
import numpy as np
import os


# traverse file recursively
def get_all_files(file_path):
    """
    :param file_path: str
    :return: ndarray[str]
    """
    filename_list = []

    for item in os.listdir(file_path):
        path = file_path + '\\' + item
        if os.path.isdir(path):     # if directory
            filename_list.extend(get_all_files(path))
        elif os.path.isfile(path):  # if file item
            filename_list.append(path)

    filename_list = np.asarray(filename_list)

    return filename_list
