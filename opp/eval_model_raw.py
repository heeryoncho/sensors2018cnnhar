import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import keras.backend as K
import select_data as sd

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code outputs the RAW sensor data HAR performance using 
1D CNN for classifying 2-class & 4-class HAR tasks, namely,
    - end2end (4-class)    # (stand, walk, sit, & lie)
    - abst (2-class)   # abstract activities (up & down)
    - up (2-class)   # up position activities (stand & walk)
    - down (2-class)   # down position activities (sit & lie)
      
See bar graph with 1D CNN in Figure 16, blue bars indicating Raw Time Series data. 
(Sensors 2018, 18(4), 1055, page 17 of 24)

'''


print "========================================================="
print "   Outputs performance of 1D CNN using RAW sensor data"
print "   On various 2-class & 4-class classification tasks:"
print "   end2end, abst, up, & down."
print "========================================================="

sensor_data_type = "raw"
model_list = ['end2end', 'abst', 'up', 'down']

for m in model_list:
    X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data(sensor_data_type, m)
    model_path = "model/{}_{}.hdf5".format(sensor_data_type, m)
    model = load_model(model_path)
    pred = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    print "\n-------------------------------------"
    print ">>> Test ACC [{}_{}]".format(sensor_data_type, m)
    print accuracy_score(y_test, np.argmax(pred, axis=1))
    print confusion_matrix(y_test, np.argmax(pred, axis=1))
    del model
    K.clear_session()


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/eval_model_raw.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
=========================================================
   Outputs performance of 1D CNN using RAW sensor data
   On various 2-class & 4-class classification tasks:
   end2end, abst, up, & down.
=========================================================

-------------------------------------
>>> Test ACC [raw_end2end]
0.931743909685
[[5193  123   10    0]
 [ 785 3099    1    0]
 [   0    0 3460    0]
 [   0    0    0  793]]

-------------------------------------
>>> Test ACC [raw_abst]
0.999925727867
[[9211    0]
 [   1 4252]]

-------------------------------------
>>> Test ACC [raw_up]
0.901747910107
[[4845  481]
 [ 424 3461]]

-------------------------------------
>>> Test ACC [raw_down]
0.997883846696
[[3460    0]
 [   9  784]]

Process finished with exit code 0
'''