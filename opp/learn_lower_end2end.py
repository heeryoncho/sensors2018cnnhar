import numpy as np
from keras.models import Model
from keras import losses
from keras.initializers import Constant, TruncatedNormal
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
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

This code learns End-to-End activity classification model using LOWER body sensors data.
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "end2end")

n_classes = 4

# Generates one-hot encoding of the activity labels
y_train_oh = np.eye(n_classes)[y_train]
y_test_oh = np.eye(n_classes)[y_test]
y_valid_oh = np.eye(n_classes)[y_valid]

# Fit 1d CNN

# Input layer
visible = Input(shape=(156, 1))

b_init = Constant(value=0.0)
k_init = TruncatedNormal(mean=0.0, stddev=0.025, seed=2017)

# First feature extractor
conv11 = Conv1D(50, kernel_size=2, strides=1, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(visible)
conv12 = Conv1D(100, kernel_size=2, strides=1, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(conv11)
conv13 = Conv1D(300, kernel_size=2, strides=1, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(conv12)
flat1 = Flatten()(conv13)

# Second feature extractor
conv21 = Conv1D(50, kernel_size=3, strides=2, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(visible)
conv22 = Conv1D(100, kernel_size=3, strides=2, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(conv21)
conv23 = Conv1D(300, kernel_size=3, strides=2, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(conv22)
flat2 = Flatten()(conv23)

# Third feature extractor
conv31 = Conv1D(50, kernel_size=4, strides=3, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(visible)
conv32 = Conv1D(100, kernel_size=4, strides=3, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(conv31)
conv33 = Conv1D(300, kernel_size=4, strides=3, activation='relu', padding='valid', bias_initializer=b_init,
                kernel_initializer=k_init)(conv32)
flat3 = Flatten()(conv33)

# Merge feature extractors
merge = concatenate([flat1, flat2, flat3])

# Dropout & fully-connected layer
do = Dropout(0.5, seed=2017)(merge)
output = Dense(4, activation='softmax', bias_initializer=b_init, kernel_initializer=k_init)(do)

# Result
model = Model(inputs=visible, outputs=output)

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

adam = Adam(lr=0.00005)
model.compile(loss=losses.squared_hinge, optimizer=adam, metrics=['accuracy'])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Comment out to disable learning /// Uncomment below to allow learning:

#model.fit(np.expand_dims(X_train, axis=2), y_train_oh, batch_size=32, epochs=5, verbose=2,
#          validation_data=(np.expand_dims(X_valid, axis=2), y_valid_oh),
#          callbacks=[cp_cb], shuffle=True)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Save 1D CNN model image
if not os.path.exists('fig/model_lower_end2end.png'):
    model_file = 'fig/model_lower_end2end.png'
    plot_model(model, to_file=model_file)

del model
K.clear_session()


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/learn_lower_end2end.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 156, 1)       0                                            
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 155, 50)      150         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 77, 50)       200         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 51, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 154, 100)     10100       conv1d_1[0][0]                   
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 38, 100)      15100       conv1d_4[0][0]                   
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 16, 100)      20100       conv1d_7[0][0]                   
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 153, 300)     60300       conv1d_2[0][0]                   
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 18, 300)      90300       conv1d_5[0][0]                   
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 5, 300)       120300      conv1d_8[0][0]                   
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 45900)        0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 5400)         0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 1500)         0           conv1d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 52800)        0           flatten_1[0][0]                  
                                                                 flatten_2[0][0]                  
                                                                 flatten_3[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 52800)        0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4)            211204      dropout_1[0][0]                  
==================================================================================================
Total params: 528,004
Trainable params: 528,004
Non-trainable params: 0
__________________________________________________________________________________________________
None

Process finished with exit code 0

'''