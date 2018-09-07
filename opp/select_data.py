import numpy as np
from scipy import ndimage


def load_data(sensor_type, class_type):
    '''
    :param sensor_type: 'raw', 'lower', 'upper'
    :param class_type: 'abst', 'up', 'down', 'end2end'
    :return: X_train, y_train, X_valid, y_valid, X_test, y_test
    '''
    dir_path = '../data/OpportunityUCIDataset/processed/'

    X_train = np.load(dir_path + "{}_train_X.npy".format(sensor_type))
    y_train = np.load(dir_path + "{}_train_y.npy".format(sensor_type))

    X_valid = np.load(dir_path + "{}_valid_X.npy".format(sensor_type))
    y_valid = np.load(dir_path + "{}_valid_y.npy".format(sensor_type))

    X_test = np.load(dir_path + "{}_test_X.npy".format(sensor_type))
    y_test = np.load(dir_path + "{}_test_y.npy".format(sensor_type))

    if class_type in ["abst", "end2end"]:
        y_train, y_valid, y_test = update_y([y_train, y_valid, y_test], class_type)
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    if class_type in ["down", "up"]:
        result = update_xny([X_train, X_valid, X_test], [y_train, y_valid, y_test], class_type)
        return result


def update_y(y_list, class_type):
    '''
    :param y_list: y_train ,y_valid, y_test
    :param class_type: 'abst', 'end2end'
    :return: y_train ,y_valid, y_test
    '''
    y_train, y_valid, y_test = y_list

    if class_type is "abst":
        y_train[y_train == 1] = 0
        y_train[y_train == 2] = 0
        y_train[y_train == 4] = 1
        y_train[y_train == 5] = 1

        y_test[y_test == 1] = 0
        y_test[y_test == 2] = 0
        y_test[y_test == 4] = 1
        y_test[y_test == 5] = 1

        y_valid[y_valid == 1] = 0
        y_valid[y_valid == 2] = 0
        y_valid[y_valid == 4] = 1
        y_valid[y_valid == 5] = 1

    if class_type is "end2end":
        y_train[y_train == 1] = 0
        y_train[y_train == 2] = 1
        y_train[y_train == 4] = 2
        y_train[y_train == 5] = 3

        y_test[y_test == 1] = 0
        y_test[y_test == 2] = 1
        y_test[y_test == 4] = 2
        y_test[y_test == 5] = 3

        y_valid[y_valid == 1] = 0
        y_valid[y_valid == 2] = 1
        y_valid[y_valid == 4] = 2
        y_valid[y_valid == 5] = 3

    return y_train, y_valid, y_test


def update_xny(X_list, y_list, class_type):
    '''
    :param X_list: X_train, X_valid, X_test
    :param y_list: y_train, y_valid, y_test
    :param class_type: 'down', 'up
    :return: X_train, y_train, X_valid, y_valid, X_test, y_test
    '''
    X_train, X_valid, X_test = X_list
    y_train, y_valid, y_test = y_list

    if class_type is "up":
        tr_stand = np.where(y_train == 1)[0]
        tr_walk = np.where(y_train == 2)[0]
        tr_up = np.concatenate([tr_stand, tr_walk])

        vd_stand = np.where(y_valid == 1)[0]
        vd_walk = np.where(y_valid == 2)[0]
        vd_up = np.concatenate([vd_stand, vd_walk])

        ts_stand = np.where(y_test == 1)[0]
        ts_walk = np.where(y_test == 2)[0]
        ts_up = np.concatenate([ts_stand, ts_walk])

        X_train_up = X_train[tr_up]
        y_train_up = y_train[tr_up]

        y_train_up[y_train_up == 1] = 0
        y_train_up[y_train_up == 2] = 1

        X_test_up = X_test[ts_up]
        y_test_up = y_test[ts_up]

        y_test_up[y_test_up == 1] = 0
        y_test_up[y_test_up == 2] = 1

        X_valid_up = X_valid[vd_up]
        y_valid_up = y_valid[vd_up]

        y_valid_up[y_valid_up == 1] = 0
        y_valid_up[y_valid_up == 2] = 1

        return X_train_up, y_train_up, X_valid_up, y_valid_up, X_test_up, y_test_up

    if class_type is "down":
        tr_sit = np.where(y_train == 4)[0]
        tr_lie = np.where(y_train == 5)[0]
        tr_down = np.concatenate([tr_sit, tr_lie])

        vd_sit = np.where(y_valid == 4)[0]
        vd_lie = np.where(y_valid == 5)[0]
        vd_down = np.concatenate([vd_sit, vd_lie])

        ts_sit = np.where(y_test == 4)[0]
        ts_lie = np.where(y_test == 5)[0]
        ts_down = np.concatenate([ts_sit, ts_lie])

        X_train_down = X_train[tr_down]
        y_train_down = y_train[tr_down]

        y_train_down[y_train_down == 4] = 0
        y_train_down[y_train_down == 5] = 1

        X_test_down = X_test[ts_down]
        y_test_down = y_test[ts_down]

        y_test_down[y_test_down == 4] = 0
        y_test_down[y_test_down == 5] = 1

        X_valid_down = X_valid[vd_down]
        y_valid_down = y_valid[vd_down]

        y_valid_down[y_valid_down == 4] = 0
        y_valid_down[y_valid_down == 5] = 1

        return X_train_down, y_train_down, X_valid_down, y_valid_down, X_test_down, y_test_down


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