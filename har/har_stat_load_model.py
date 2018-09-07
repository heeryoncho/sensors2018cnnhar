import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import process_data

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code outputs the result of [UCI HAR (Static) : 1D CNN Only (%) : 96.60%]
given in Table 8. (Sensors 2018, 18(4), 1055, page 19 of 24)

The HAR model is saved as model/static.hdf5

'''

# Load all train and test data (* dynamic and static data are mixed.)

X_train_all = process_data.load_x("train")   # at this stage, the data includes both dynamic and static HAR data
y_train_all = process_data.load_y("train")

X_test_all = process_data.load_x("test")
y_test_all = process_data.load_y("test")

# --------------------------------------
# Only static HAR data are selected
# --------------------------------------

# Select static HAR train data

static_1 = np.where(y_train_all == 4)[0]
static_2 = np.where(y_train_all == 5)[0]
static_3 = np.where(y_train_all == 6)[0]
static = np.concatenate([static_1, static_2, static_3])

X_train = X_train_all[static]
y_train = y_train_all[static]

# Convert (4, 5, 6) labels to (0, 1, 2)
y_train  = y_train - 4

print "\n+++ DATA STATISTICS +++\n"
print "train_static shape: ", X_train.shape

# Select static HAR test data

static_1 = np.where(y_test_all == 4)[0]
static_2 = np.where(y_test_all == 5)[0]
static_3 = np.where(y_test_all == 6)[0]
static = np.concatenate([static_1, static_2, static_3])

X_test = X_test_all[static]
y_test = y_test_all[static]

# Convert (4, 5, 6) labels to (0, 1, 2)
y_test  = y_test - 4

print "test_static shape: ", X_test.shape


# Display static model accuracy

print "\n+++ STATIC MODEL ACCURACY (See Table 8 in paper) +++\n"

model_path = "model/static.hdf5"
model = load_model(model_path)

pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
print "------ TRAIN ACCURACY ------"
print accuracy_score(y_train, np.argmax(pred_train, axis=1))
print confusion_matrix(y_train, np.argmax(pred_train, axis=1))

pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
print "------ TEST ACCURACY ------"
print accuracy_score(y_test, np.argmax(pred_test, axis=1))
print confusion_matrix(y_test, np.argmax(pred_test, axis=1))


'''

/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/har/har_stat_load_model.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

+++ DATA STATISTICS +++

train_static shape:  (4067, 561)
test_static shape:  (1560, 561)

+++ STATIC MODEL ACCURACY (See Table 8 in paper) +++

------ TRAIN ACCURACY ------
0.992377673961
[[1275   11    0]
 [  20 1354    0]
 [   0    0 1407]]
------ TEST ACCURACY ------
0.966025641026
[[453  38   0]
 [ 14 518   0]
 [  1   0 536]]

Process finished with exit code 0

'''