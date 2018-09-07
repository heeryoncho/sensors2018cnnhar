import numpy as np
from scipy import ndimage


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code loads and sharpens UCI HAR Dataset data.

UCI HAR Dataset data can be downloaded from:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Unzipped dataset should be placed inside the '../data/UCI HAR Dataset/' folder.

'''

dir_path = '../data/UCI HAR Dataset/'


def load_x(train_or_test):
    global dir_path
    if train_or_test is "train":
        x_path = dir_path + 'train/X_train.txt'
    elif train_or_test is "test":
        x_path = dir_path + 'test/X_test.txt'

    with open(x_path) as f:
        container = f.readlines()

    result = []
    for line in container:
        tmp1 = line.strip()
        tmp2 = tmp1.replace('  ', ' ')     # removes inconsistent blank spaces
        tmp_ary = map(float, tmp2.split(' '))
        result.append(tmp_ary)
    return np.array(result)


def load_y(train_or_test):
    global dir_path
    if train_or_test is "train":
        y_path = dir_path + 'train/y_train.txt'
    elif train_or_test is "test":
        y_path = dir_path + 'test/y_test.txt'

    with open(y_path) as f:
        container = f.readlines()

    result = []
    for line in container:
        num_str = line.strip()
        result.append(int(num_str))
    return np.array(result)


def sharpen(x_test, sigma, alpha):
    r = x_test.shape[0]
    c = x_test.shape[1]
    container = np.empty((r, c))
    i = 0

    for row in x_test:
        test = np.array([row])
        blurred = ndimage.gaussian_filter(test, sigma)
        sharpened = test + alpha * (test - blurred)
        container[i] = sharpened
        i = i + 1
    return container
