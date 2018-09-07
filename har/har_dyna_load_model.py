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

This code outputs the result of [UCI HAR (Dynamic) : 1D CNN Only (%) : 97.98%]
given in Table 8. (Sensors 2018, 18(4), 1055, page 19 of 24)

The HAR model is saved as model/dynamic.hdf5

'''

# Load all train and test data (* dynamic and static data are mixed.)

X_train_all = process_data.load_x("train")   # at this stage, the data includes both dynamic and static HAR data
y_train_all = process_data.load_y("train")

X_test_all = process_data.load_x("test")
y_test_all = process_data.load_y("test")

# --------------------------------------
# Only dynamic HAR data are selected
# --------------------------------------

# Select dynamic HAR train data

dynamic_1 = np.where(y_train_all == 1)[0]
dynamic_2 = np.where(y_train_all == 2)[0]
dynamic_3 = np.where(y_train_all == 3)[0]
dynamic = np.concatenate([dynamic_1, dynamic_2, dynamic_3])

X_train = X_train_all[dynamic]
y_train = y_train_all[dynamic]

# Convert (1, 2, 3) labels to (0, 1, 2)
y_train  = y_train - 1

print "\n+++ DATA STATISTICS +++\n"
print "train_dynamic shape: ", X_train.shape

# Select dynamic HAR test data

dynamic_1 = np.where(y_test_all == 1)[0]
dynamic_2 = np.where(y_test_all == 2)[0]
dynamic_3 = np.where(y_test_all == 3)[0]
dynamic = np.concatenate([dynamic_1, dynamic_2, dynamic_3])

X_test = X_test_all[dynamic]
y_test = y_test_all[dynamic]

# Convert (1, 2, 3) labels to (0, 1, 2)
y_test  = y_test - 1

print "test_dynamic shape: ", X_test.shape


# Display dynamic model accuracy

print "\n+++ DYNAMIC MODEL ACCURACY (See Table 8 in paper) +++\n"

model_path = "model/dynamic.hdf5"
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

/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/har/har_dyna_load_model.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

+++ DATA STATISTICS +++

train_dynamic shape:  (3285, 561)
test_dynamic shape:  (1387, 561)

+++ DYNAMIC MODEL ACCURACY (See Table 8 in paper) +++

------ TRAIN ACCURACY ------
0.986301369863
[[1223    3    0]
 [   4 1038   31]
 [   0    7  979]]
------ TEST ACCURACY ------
0.979812545061
[[495   0   1]
 [  2 469   0]
 [  2  23 395]]

Process finished with exit code 0

'''