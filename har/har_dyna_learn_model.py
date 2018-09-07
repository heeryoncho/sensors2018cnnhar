import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import random
from numpy.random import seed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import keras.backend as K
import process_data


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code learns dynamic HAR model given in Figure 10.
(Sensors 2018, 18(4), 1055, page 13 of 24)

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
dynamic_list = dynamic.tolist()

# Shuffle dynamic data index
r = random.random()
random.shuffle(dynamic_list, lambda: r)

dynamic = np.array(dynamic_list)

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

n_classes = 3

# Convert to one hot encoding vector
y_train_dynamic_oh = np.eye(n_classes)[y_train]

# Fit 1d CNN for dynamic HAR

seed(2017)
model = Sequential()
model.add(Conv1D(100, 3, input_shape=(561, 1), activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.add(Dropout(0.5))

adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Summarize layers
print(model.summary())

# Save model image
if not os.path.exists('fig_har_dyna.png'):
    model_file = 'fig_har_dyna.png'
    plot_model(model, to_file=model_file)

new_dir = 'model/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
fpath = new_dir + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'

cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# To disable learning, the below code - two lines - is commented.
# To enable learning uncomment the below two lines of code.

#model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
#          batch_size=32, epochs=50, verbose=2, validation_split=0.2, callbacks=[cp_cb])
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

del model
K.clear_session()



'''

/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/har/har_dyna_learn_model.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

+++ DATA STATISTICS +++

train_dynamic shape:  (3285, 561)
test_dynamic shape:  (1387, 561)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 559, 100)          400       
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 186, 100)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 18600)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 55803     
_________________________________________________________________
dropout_1 (Dropout)          (None, 3)                 0         
=================================================================
Total params: 56,203
Trainable params: 56,203
Non-trainable params: 0
_________________________________________________________________
None


Process finished with exit code 0

'''