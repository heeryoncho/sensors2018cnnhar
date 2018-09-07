import numpy as np
from numpy.random import seed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.initializers import TruncatedNormal
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import keras.backend as K
import select_data as sd

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code learns UP position activity classification model using RAW (lower) sensor data.
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("raw", "up")

n_classes = 2

# Generates one-hot encoding of the activity labels
y_train_oh = np.eye(n_classes)[y_train]
y_test_oh = np.eye(n_classes)[y_test]
y_valid_oh = np.eye(n_classes)[y_valid]

# Fit 1d CNN
k_init = TruncatedNormal(mean=0.0, stddev=0.025, seed=2017)

seed(2017)
model = Sequential()
model.add(Conv1D(100, 3, input_shape=(585, 1), activation='relu', kernel_initializer=k_init))
model.add(MaxPooling1D(3, strides=1))
model.add(Conv1D(500, 3, activation='relu', kernel_initializer=k_init))
model.add(Flatten())
model.add(Dense(2, activation='softmax', kernel_initializer='uniform'))
model.add(Dropout(0.33))

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Summarize layers
print(model.summary())

model_dir = 'model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

fig_dir = 'fig/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

fpath = model_dir + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'

cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Comment out to disable learning /// Uncomment below to enable learning:

#model.fit(np.expand_dims(X_train, axis=2), y_train_oh,
#          validation_data=(np.expand_dims(X_valid, axis=2), y_valid_oh),
#          batch_size=32, epochs=5, verbose=2, callbacks=[cp_cb])
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Save 1D CNN model image
if not os.path.exists('fig/model_raw_up.png'):
    model_file = 'fig/model_raw_up.png'
    plot_model(model, to_file=model_file)

del model
K.clear_session()


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/learn_raw_up.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 583, 100)          400       
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 581, 100)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 579, 500)          150500    
_________________________________________________________________
flatten_1 (Flatten)          (None, 289500)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 579002    
_________________________________________________________________
dropout_1 (Dropout)          (None, 2)                 0         
=================================================================
Total params: 729,902
Trainable params: 729,902
Non-trainable params: 0
_________________________________________________________________
None

Process finished with exit code 0
'''