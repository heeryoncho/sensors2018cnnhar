import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import random
from numpy.random import seed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import keras.backend as K
import process_data


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code learns dynamic HAR model given in Figure 11.
(Sensors 2018, 18(4), 1055, page 13 of 24)

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
static_list = static.tolist()

# Shuffle static data index
r = random.random()
random.shuffle(static_list, lambda: r)

static = np.array(static_list)

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

n_classes = 3

# Convert to one hot encoding vector
y_train_static_oh = np.eye(n_classes)[y_train]

# Fit 1d CNN for static HAR

seed(2017)
model = Sequential()
model.add(Conv1D(30, 3, input_shape=(561, 1), activation='relu'))
model.add(Conv1D(50, 3, activation='relu'))
model.add(Conv1D(100, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.add(Dropout(0.50))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Summarize layers
print(model.summary())

if not os.path.exists('fig_har_stat.png'):
    model_file = 'fig_har_stat.png'
    plot_model(model, to_file=model_file)

new_dir = 'model/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
fpath = new_dir + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'

cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# To disable learning, the below code - two lines - is commented.
# To enable learning uncomment the below two lines of code.

model.fit(np.expand_dims(X_train, axis=2), y_train_static_oh,
          batch_size=32, epochs=100, verbose=2, validation_split=0.2, callbacks=[cp_cb])
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

del model
K.clear_session()



'''

/usr/bin/python2.7 /home/hcilab/PycharmProjects/Sensors_180323/har/har_stat_learn_model.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
train_static shape:  (4067, 561)
test_static shape:  (1560, 561)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d_1 (Conv1D)            (None, 559, 30)           120       
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 557, 50)           4550      
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 555, 100)          15100     
_________________________________________________________________
flatten_1 (Flatten)          (None, 55500)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 166503    
_________________________________________________________________
dropout_1 (Dropout)          (None, 3)                 0         
=================================================================
Total params: 186,273
Trainable params: 186,273
Non-trainable params: 0
_________________________________________________________________
None
Train on 3253 samples, validate on 814 samples
Epoch 1/100
 - 2s - loss: 0.3049 - acc: 0.4384 - val_loss: 0.1512 - val_acc: 0.8305

Epoch 00001: val_loss improved from inf to 0.15120, saving model to stat/repro/weights.01-0.83.hdf5
Epoch 2/100
 - 1s - loss: 0.2588 - acc: 0.5164 - val_loss: 0.1046 - val_acc: 0.8538

Epoch 00002: val_loss improved from 0.15120 to 0.10463, saving model to stat/repro/weights.02-0.85.hdf5
Epoch 3/100
 - 1s - loss: 0.2476 - acc: 0.5238 - val_loss: 0.0960 - val_acc: 0.8636

Epoch 00003: val_loss improved from 0.10463 to 0.09599, saving model to stat/repro/weights.03-0.86.hdf5
Epoch 4/100
 - 1s - loss: 0.2496 - acc: 0.5131 - val_loss: 0.0884 - val_acc: 0.9103

Epoch 00004: val_loss improved from 0.09599 to 0.08840, saving model to stat/repro/weights.04-0.91.hdf5
Epoch 5/100
 - 1s - loss: 0.2429 - acc: 0.5223 - val_loss: 0.0857 - val_acc: 0.9287

Epoch 00005: val_loss improved from 0.08840 to 0.08574, saving model to stat/repro/weights.05-0.93.hdf5
Epoch 6/100
 - 1s - loss: 0.2375 - acc: 0.5395 - val_loss: 0.0870 - val_acc: 0.9337

Epoch 00006: val_loss did not improve
Epoch 7/100
 - 1s - loss: 0.2390 - acc: 0.5229 - val_loss: 0.0816 - val_acc: 0.9312

Epoch 00007: val_loss improved from 0.08574 to 0.08156, saving model to stat/repro/weights.07-0.93.hdf5
Epoch 8/100
 - 1s - loss: 0.2326 - acc: 0.5460 - val_loss: 0.0785 - val_acc: 0.9435

Epoch 00008: val_loss improved from 0.08156 to 0.07850, saving model to stat/repro/weights.08-0.94.hdf5
Epoch 9/100
 - 1s - loss: 0.2301 - acc: 0.5457 - val_loss: 0.0789 - val_acc: 0.9472

Epoch 00009: val_loss did not improve
Epoch 10/100
 - 1s - loss: 0.2359 - acc: 0.5297 - val_loss: 0.0754 - val_acc: 0.9496

Epoch 00010: val_loss improved from 0.07850 to 0.07541, saving model to stat/repro/weights.10-0.95.hdf5
Epoch 11/100
 - 1s - loss: 0.2291 - acc: 0.5536 - val_loss: 0.0783 - val_acc: 0.9545

Epoch 00011: val_loss did not improve
Epoch 12/100
 - 1s - loss: 0.2346 - acc: 0.5380 - val_loss: 0.0752 - val_acc: 0.9496

Epoch 00012: val_loss improved from 0.07541 to 0.07515, saving model to stat/repro/weights.12-0.95.hdf5
Epoch 13/100
 - 1s - loss: 0.2321 - acc: 0.5361 - val_loss: 0.0744 - val_acc: 0.9558

Epoch 00013: val_loss improved from 0.07515 to 0.07441, saving model to stat/repro/weights.13-0.96.hdf5
Epoch 14/100
 - 1s - loss: 0.2331 - acc: 0.5318 - val_loss: 0.0730 - val_acc: 0.9582

Epoch 00014: val_loss improved from 0.07441 to 0.07302, saving model to stat/repro/weights.14-0.96.hdf5
Epoch 15/100
 - 1s - loss: 0.2267 - acc: 0.5567 - val_loss: 0.0783 - val_acc: 0.9582

Epoch 00015: val_loss did not improve
Epoch 16/100
 - 1s - loss: 0.2332 - acc: 0.5352 - val_loss: 0.0748 - val_acc: 0.9521

Epoch 00016: val_loss did not improve
Epoch 17/100
 - 1s - loss: 0.2277 - acc: 0.5444 - val_loss: 0.0737 - val_acc: 0.9619

Epoch 00017: val_loss did not improve
Epoch 18/100
 - 1s - loss: 0.2346 - acc: 0.5321 - val_loss: 0.0736 - val_acc: 0.9681

Epoch 00018: val_loss did not improve
Epoch 19/100
 - 1s - loss: 0.2333 - acc: 0.5327 - val_loss: 0.0754 - val_acc: 0.9570

Epoch 00019: val_loss did not improve
Epoch 20/100
 - 1s - loss: 0.2307 - acc: 0.5426 - val_loss: 0.0713 - val_acc: 0.9619

Epoch 00020: val_loss improved from 0.07302 to 0.07127, saving model to stat/repro/weights.20-0.96.hdf5
Epoch 21/100
 - 1s - loss: 0.2304 - acc: 0.5463 - val_loss: 0.0743 - val_acc: 0.9558

Epoch 00021: val_loss did not improve
Epoch 22/100
 - 1s - loss: 0.2314 - acc: 0.5327 - val_loss: 0.0750 - val_acc: 0.9619

Epoch 00022: val_loss did not improve
Epoch 23/100
 - 1s - loss: 0.2323 - acc: 0.5321 - val_loss: 0.0743 - val_acc: 0.9558

Epoch 00023: val_loss did not improve
Epoch 24/100
 - 1s - loss: 0.2380 - acc: 0.5152 - val_loss: 0.0730 - val_acc: 0.9644

Epoch 00024: val_loss did not improve
Epoch 25/100
 - 1s - loss: 0.2352 - acc: 0.5260 - val_loss: 0.0715 - val_acc: 0.9595

Epoch 00025: val_loss did not improve
Epoch 26/100
 - 1s - loss: 0.2285 - acc: 0.5395 - val_loss: 0.0724 - val_acc: 0.9644

Epoch 00026: val_loss did not improve
Epoch 27/100
 - 1s - loss: 0.2321 - acc: 0.5337 - val_loss: 0.0708 - val_acc: 0.9656

Epoch 00027: val_loss improved from 0.07127 to 0.07077, saving model to stat/repro/weights.27-0.97.hdf5
Epoch 28/100
 - 1s - loss: 0.2316 - acc: 0.5395 - val_loss: 0.0708 - val_acc: 0.9705

Epoch 00028: val_loss improved from 0.07077 to 0.07077, saving model to stat/repro/weights.28-0.97.hdf5
Epoch 29/100
 - 1s - loss: 0.2292 - acc: 0.5398 - val_loss: 0.0724 - val_acc: 0.9705

Epoch 00029: val_loss did not improve
Epoch 30/100
 - 1s - loss: 0.2289 - acc: 0.5398 - val_loss: 0.0726 - val_acc: 0.9681

Epoch 00030: val_loss did not improve
Epoch 31/100
 - 1s - loss: 0.2293 - acc: 0.5398 - val_loss: 0.0739 - val_acc: 0.9668

Epoch 00031: val_loss did not improve
Epoch 32/100
 - 1s - loss: 0.2294 - acc: 0.5460 - val_loss: 0.0693 - val_acc: 0.9693

Epoch 00032: val_loss improved from 0.07077 to 0.06932, saving model to stat/repro/weights.32-0.97.hdf5
Epoch 33/100
 - 1s - loss: 0.2293 - acc: 0.5358 - val_loss: 0.0712 - val_acc: 0.9607

Epoch 00033: val_loss did not improve
Epoch 34/100
 - 1s - loss: 0.2300 - acc: 0.5426 - val_loss: 0.0742 - val_acc: 0.9656

Epoch 00034: val_loss did not improve
Epoch 35/100
 - 1s - loss: 0.2333 - acc: 0.5275 - val_loss: 0.0689 - val_acc: 0.9681

Epoch 00035: val_loss improved from 0.06932 to 0.06887, saving model to stat/repro/weights.35-0.97.hdf5
Epoch 36/100
 - 1s - loss: 0.2285 - acc: 0.5429 - val_loss: 0.0695 - val_acc: 0.9693

Epoch 00036: val_loss did not improve
Epoch 37/100
 - 1s - loss: 0.2267 - acc: 0.5475 - val_loss: 0.0718 - val_acc: 0.9730

Epoch 00037: val_loss did not improve
Epoch 38/100
 - 1s - loss: 0.2287 - acc: 0.5386 - val_loss: 0.0722 - val_acc: 0.9693

Epoch 00038: val_loss did not improve
Epoch 39/100
 - 1s - loss: 0.2307 - acc: 0.5315 - val_loss: 0.0761 - val_acc: 0.9656

Epoch 00039: val_loss did not improve
Epoch 40/100
 - 1s - loss: 0.2298 - acc: 0.5457 - val_loss: 0.0687 - val_acc: 0.9730

Epoch 00040: val_loss improved from 0.06887 to 0.06872, saving model to stat/repro/weights.40-0.97.hdf5
Epoch 41/100
 - 1s - loss: 0.2335 - acc: 0.5257 - val_loss: 0.0733 - val_acc: 0.9693

Epoch 00041: val_loss did not improve
Epoch 42/100
 - 1s - loss: 0.2285 - acc: 0.5407 - val_loss: 0.0720 - val_acc: 0.9693

Epoch 00042: val_loss did not improve
Epoch 43/100
 - 1s - loss: 0.2342 - acc: 0.5244 - val_loss: 0.0727 - val_acc: 0.9681

Epoch 00043: val_loss did not improve
Epoch 44/100
 - 1s - loss: 0.2241 - acc: 0.5613 - val_loss: 0.0741 - val_acc: 0.9730

Epoch 00044: val_loss did not improve
Epoch 45/100
 - 1s - loss: 0.2290 - acc: 0.5374 - val_loss: 0.0737 - val_acc: 0.9717

Epoch 00045: val_loss did not improve
Epoch 46/100
 - 1s - loss: 0.2273 - acc: 0.5420 - val_loss: 0.0754 - val_acc: 0.9705

Epoch 00046: val_loss did not improve
Epoch 47/100
 - 1s - loss: 0.2265 - acc: 0.5503 - val_loss: 0.0715 - val_acc: 0.9717

Epoch 00047: val_loss did not improve
Epoch 48/100
 - 1s - loss: 0.2303 - acc: 0.5377 - val_loss: 0.0686 - val_acc: 0.9656

Epoch 00048: val_loss improved from 0.06872 to 0.06858, saving model to stat/repro/weights.48-0.97.hdf5
Epoch 49/100
 - 1s - loss: 0.2332 - acc: 0.5337 - val_loss: 0.0726 - val_acc: 0.9681

Epoch 00049: val_loss did not improve
Epoch 50/100
 - 1s - loss: 0.2270 - acc: 0.5466 - val_loss: 0.0693 - val_acc: 0.9754

Epoch 00050: val_loss did not improve
Epoch 51/100
 - 1s - loss: 0.2309 - acc: 0.5364 - val_loss: 0.0698 - val_acc: 0.9754

Epoch 00051: val_loss did not improve
Epoch 52/100
 - 1s - loss: 0.2333 - acc: 0.5263 - val_loss: 0.0683 - val_acc: 0.9767

Epoch 00052: val_loss improved from 0.06858 to 0.06834, saving model to stat/repro/weights.52-0.98.hdf5
Epoch 53/100
 - 1s - loss: 0.2243 - acc: 0.5530 - val_loss: 0.0702 - val_acc: 0.9693

Epoch 00053: val_loss did not improve
Epoch 54/100
 - 1s - loss: 0.2328 - acc: 0.5364 - val_loss: 0.0717 - val_acc: 0.9730

Epoch 00054: val_loss did not improve
Epoch 55/100
 - 1s - loss: 0.2273 - acc: 0.5392 - val_loss: 0.0697 - val_acc: 0.9742

Epoch 00055: val_loss did not improve
Epoch 56/100
 - 1s - loss: 0.2248 - acc: 0.5601 - val_loss: 0.0726 - val_acc: 0.9717

Epoch 00056: val_loss did not improve
Epoch 57/100
 - 1s - loss: 0.2300 - acc: 0.5380 - val_loss: 0.0701 - val_acc: 0.9730

Epoch 00057: val_loss did not improve
Epoch 58/100
 - 1s - loss: 0.2313 - acc: 0.5291 - val_loss: 0.0687 - val_acc: 0.9705

Epoch 00058: val_loss did not improve
Epoch 59/100
 - 1s - loss: 0.2248 - acc: 0.5540 - val_loss: 0.0716 - val_acc: 0.9730

Epoch 00059: val_loss did not improve
Epoch 60/100
 - 1s - loss: 0.2265 - acc: 0.5389 - val_loss: 0.0685 - val_acc: 0.9742

Epoch 00060: val_loss did not improve
Epoch 61/100
 - 1s - loss: 0.2338 - acc: 0.5269 - val_loss: 0.0724 - val_acc: 0.9791

Epoch 00061: val_loss did not improve
Epoch 62/100
 - 1s - loss: 0.2287 - acc: 0.5355 - val_loss: 0.0704 - val_acc: 0.9816

Epoch 00062: val_loss did not improve
Epoch 63/100
 - 1s - loss: 0.2255 - acc: 0.5552 - val_loss: 0.0718 - val_acc: 0.9767

Epoch 00063: val_loss did not improve
Epoch 64/100
 - 1s - loss: 0.2260 - acc: 0.5463 - val_loss: 0.0737 - val_acc: 0.9779

Epoch 00064: val_loss did not improve
Epoch 65/100
 - 1s - loss: 0.2242 - acc: 0.5493 - val_loss: 0.0696 - val_acc: 0.9803

Epoch 00065: val_loss did not improve
Epoch 66/100
 - 1s - loss: 0.2285 - acc: 0.5438 - val_loss: 0.0698 - val_acc: 0.9791

Epoch 00066: val_loss did not improve
Epoch 67/100
 - 1s - loss: 0.2377 - acc: 0.5097 - val_loss: 0.0690 - val_acc: 0.9779

Epoch 00067: val_loss did not improve
Epoch 68/100
 - 1s - loss: 0.2277 - acc: 0.5346 - val_loss: 0.0734 - val_acc: 0.9791

Epoch 00068: val_loss did not improve
Epoch 69/100
 - 1s - loss: 0.2298 - acc: 0.5367 - val_loss: 0.0683 - val_acc: 0.9816

Epoch 00069: val_loss improved from 0.06834 to 0.06827, saving model to stat/repro/weights.69-0.98.hdf5
Epoch 70/100
 - 1s - loss: 0.2279 - acc: 0.5453 - val_loss: 0.0697 - val_acc: 0.9816

Epoch 00070: val_loss did not improve
Epoch 71/100
 - 1s - loss: 0.2291 - acc: 0.5389 - val_loss: 0.0729 - val_acc: 0.9779

Epoch 00071: val_loss did not improve
Epoch 72/100
 - 1s - loss: 0.2291 - acc: 0.5398 - val_loss: 0.0693 - val_acc: 0.9791

Epoch 00072: val_loss did not improve
Epoch 73/100
 - 1s - loss: 0.2304 - acc: 0.5346 - val_loss: 0.0682 - val_acc: 0.9816

Epoch 00073: val_loss improved from 0.06827 to 0.06824, saving model to stat/repro/weights.73-0.98.hdf5
Epoch 74/100
 - 1s - loss: 0.2282 - acc: 0.5401 - val_loss: 0.0674 - val_acc: 0.9803

Epoch 00074: val_loss improved from 0.06824 to 0.06739, saving model to stat/repro/weights.74-0.98.hdf5
Epoch 75/100
 - 1s - loss: 0.2279 - acc: 0.5380 - val_loss: 0.0704 - val_acc: 0.9828

Epoch 00075: val_loss did not improve
Epoch 76/100
 - 1s - loss: 0.2295 - acc: 0.5346 - val_loss: 0.0683 - val_acc: 0.9803

Epoch 00076: val_loss did not improve
Epoch 77/100
 - 1s - loss: 0.2301 - acc: 0.5272 - val_loss: 0.0725 - val_acc: 0.9803

Epoch 00077: val_loss did not improve
Epoch 78/100
 - 1s - loss: 0.2275 - acc: 0.5450 - val_loss: 0.0712 - val_acc: 0.9779

Epoch 00078: val_loss did not improve
Epoch 79/100
 - 1s - loss: 0.2288 - acc: 0.5355 - val_loss: 0.0678 - val_acc: 0.9803

Epoch 00079: val_loss did not improve
Epoch 80/100
 - 1s - loss: 0.2294 - acc: 0.5386 - val_loss: 0.0668 - val_acc: 0.9791

Epoch 00080: val_loss improved from 0.06739 to 0.06683, saving model to stat/repro/weights.80-0.98.hdf5
Epoch 81/100
 - 1s - loss: 0.2300 - acc: 0.5272 - val_loss: 0.0687 - val_acc: 0.9754

Epoch 00081: val_loss did not improve
Epoch 82/100
 - 1s - loss: 0.2280 - acc: 0.5441 - val_loss: 0.0697 - val_acc: 0.9779

Epoch 00082: val_loss did not improve
Epoch 83/100
 - 1s - loss: 0.2270 - acc: 0.5407 - val_loss: 0.0681 - val_acc: 0.9803

Epoch 00083: val_loss did not improve
Epoch 84/100
 - 1s - loss: 0.2272 - acc: 0.5420 - val_loss: 0.0653 - val_acc: 0.9803

Epoch 00084: val_loss improved from 0.06683 to 0.06527, saving model to stat/repro/weights.84-0.98.hdf5
Epoch 85/100
 - 1s - loss: 0.2288 - acc: 0.5321 - val_loss: 0.0688 - val_acc: 0.9754

Epoch 00085: val_loss did not improve
Epoch 86/100
 - 1s - loss: 0.2245 - acc: 0.5472 - val_loss: 0.0703 - val_acc: 0.9816

Epoch 00086: val_loss did not improve
Epoch 87/100
 - 1s - loss: 0.2325 - acc: 0.5287 - val_loss: 0.0706 - val_acc: 0.9828

Epoch 00087: val_loss did not improve
Epoch 88/100
 - 1s - loss: 0.2305 - acc: 0.5352 - val_loss: 0.0709 - val_acc: 0.9840

Epoch 00088: val_loss did not improve
Epoch 89/100
 - 1s - loss: 0.2267 - acc: 0.5420 - val_loss: 0.0679 - val_acc: 0.9816

Epoch 00089: val_loss did not improve
Epoch 90/100
 - 1s - loss: 0.2291 - acc: 0.5343 - val_loss: 0.0691 - val_acc: 0.9853

Epoch 00090: val_loss did not improve
Epoch 91/100
 - 1s - loss: 0.2285 - acc: 0.5374 - val_loss: 0.0700 - val_acc: 0.9816

Epoch 00091: val_loss did not improve
Epoch 92/100
 - 1s - loss: 0.2302 - acc: 0.5370 - val_loss: 0.0684 - val_acc: 0.9816

Epoch 00092: val_loss did not improve
Epoch 93/100
 - 1s - loss: 0.2320 - acc: 0.5229 - val_loss: 0.0669 - val_acc: 0.9840

Epoch 00093: val_loss did not improve
Epoch 94/100
 - 1s - loss: 0.2245 - acc: 0.5496 - val_loss: 0.0722 - val_acc: 0.9828

Epoch 00094: val_loss did not improve
Epoch 95/100
 - 1s - loss: 0.2308 - acc: 0.5269 - val_loss: 0.0720 - val_acc: 0.9853

Epoch 00095: val_loss did not improve
Epoch 96/100
 - 1s - loss: 0.2231 - acc: 0.5521 - val_loss: 0.0724 - val_acc: 0.9816

Epoch 00096: val_loss did not improve
Epoch 97/100
 - 1s - loss: 0.2301 - acc: 0.5330 - val_loss: 0.0670 - val_acc: 0.9840

Epoch 00097: val_loss did not improve
Epoch 98/100
 - 1s - loss: 0.2279 - acc: 0.5358 - val_loss: 0.0682 - val_acc: 0.9853

Epoch 00098: val_loss did not improve
Epoch 99/100
 - 1s - loss: 0.2198 - acc: 0.5672 - val_loss: 0.0652 - val_acc: 0.9828

Epoch 00099: val_loss improved from 0.06527 to 0.06515, saving model to stat/repro/weights.99-0.98.hdf5
Epoch 100/100
 - 1s - loss: 0.2291 - acc: 0.5303 - val_loss: 0.0697 - val_acc: 0.9840

Epoch 00100: val_loss did not improve

Process finished with exit code 0



===============================================================
In Paper
===============================================================

/usr/bin/python2.7 /home/hcilab/PycharmProjects/HAR/UCI_HAR/0713_har_static_7.py
Using TensorFlow backend.
train_static shape:  (4067, 561)
test_static shape:  (1560, 561)
(None, 560, 30)
(None, 559, 50)
(None, 558, 100)
(None, 55800)
(None, 559, 30)
(None, 557, 50)
(None, 555, 100)
(None, 55500)
(None, 3)
Train on 3253 samples, validate on 814 samples
Epoch 1/100
Epoch 00000: val_loss improved from inf to 0.21386, saving model to static_7/weights.00-0.55.hdf5
1s - loss: 0.3297 - acc: 0.3584 - val_loss: 0.2139 - val_acc: 0.5541
Epoch 2/100
Epoch 00001: val_loss improved from 0.21386 to 0.09936, saving model to static_7/weights.01-0.92.hdf5
0s - loss: 0.2780 - acc: 0.4820 - val_loss: 0.0994 - val_acc: 0.9165
Epoch 3/100
Epoch 00002: val_loss improved from 0.09936 to 0.09136, saving model to static_7/weights.02-0.88.hdf5
0s - loss: 0.2599 - acc: 0.5020 - val_loss: 0.0914 - val_acc: 0.8784
Epoch 4/100
Epoch 00003: val_loss improved from 0.09136 to 0.08976, saving model to static_7/weights.03-0.88.hdf5
0s - loss: 0.2477 - acc: 0.5238 - val_loss: 0.0898 - val_acc: 0.8845
Epoch 5/100
Epoch 00004: val_loss improved from 0.08976 to 0.07881, saving model to static_7/weights.04-0.94.hdf5
0s - loss: 0.2434 - acc: 0.5257 - val_loss: 0.0788 - val_acc: 0.9361
Epoch 6/100
Epoch 00005: val_loss did not improve
0s - loss: 0.2425 - acc: 0.5269 - val_loss: 0.0836 - val_acc: 0.9349
Epoch 7/100
Epoch 00006: val_loss improved from 0.07881 to 0.07131, saving model to static_7/weights.06-0.95.hdf5
0s - loss: 0.2457 - acc: 0.5140 - val_loss: 0.0713 - val_acc: 0.9533
Epoch 8/100
Epoch 00007: val_loss did not improve
0s - loss: 0.2386 - acc: 0.5297 - val_loss: 0.0731 - val_acc: 0.9644
Epoch 9/100
Epoch 00008: val_loss improved from 0.07131 to 0.06983, saving model to static_7/weights.08-0.97.hdf5
0s - loss: 0.2416 - acc: 0.5192 - val_loss: 0.0698 - val_acc: 0.9656
Epoch 10/100
Epoch 00009: val_loss did not improve
0s - loss: 0.2393 - acc: 0.5275 - val_loss: 0.0705 - val_acc: 0.9631
Epoch 11/100
Epoch 00010: val_loss improved from 0.06983 to 0.06256, saving model to static_7/weights.10-0.98.hdf5
0s - loss: 0.2387 - acc: 0.5269 - val_loss: 0.0626 - val_acc: 0.9816
Epoch 12/100
Epoch 00011: val_loss did not improve
0s - loss: 0.2343 - acc: 0.5361 - val_loss: 0.0655 - val_acc: 0.9730
Epoch 13/100
Epoch 00012: val_loss did not improve
0s - loss: 0.2359 - acc: 0.5330 - val_loss: 0.0650 - val_acc: 0.9816
Epoch 14/100
Epoch 00013: val_loss did not improve
0s - loss: 0.2341 - acc: 0.5318 - val_loss: 0.0684 - val_acc: 0.9742
Epoch 15/100
Epoch 00014: val_loss did not improve
0s - loss: 0.2322 - acc: 0.5413 - val_loss: 0.0741 - val_acc: 0.9509
Epoch 16/100
Epoch 00015: val_loss did not improve
0s - loss: 0.2370 - acc: 0.5244 - val_loss: 0.0693 - val_acc: 0.9631
Epoch 17/100
Epoch 00016: val_loss did not improve
0s - loss: 0.2348 - acc: 0.5321 - val_loss: 0.0738 - val_acc: 0.9681
Epoch 18/100
Epoch 00017: val_loss did not improve
0s - loss: 0.2339 - acc: 0.5349 - val_loss: 0.0710 - val_acc: 0.9717
Epoch 19/100
Epoch 00018: val_loss did not improve
0s - loss: 0.2339 - acc: 0.5367 - val_loss: 0.0666 - val_acc: 0.9619
Epoch 20/100
Epoch 00019: val_loss did not improve
0s - loss: 0.2305 - acc: 0.5420 - val_loss: 0.0640 - val_acc: 0.9816
Epoch 21/100
Epoch 00020: val_loss did not improve
0s - loss: 0.2389 - acc: 0.5161 - val_loss: 0.0662 - val_acc: 0.9791
Epoch 22/100
Epoch 00021: val_loss improved from 0.06256 to 0.06014, saving model to static_7/weights.21-0.99.hdf5
0s - loss: 0.2308 - acc: 0.5374 - val_loss: 0.0601 - val_acc: 0.9889
Epoch 23/100
Epoch 00022: val_loss did not improve
0s - loss: 0.2365 - acc: 0.5226 - val_loss: 0.0614 - val_acc: 0.9853
Epoch 24/100
Epoch 00023: val_loss did not improve
0s - loss: 0.2353 - acc: 0.5244 - val_loss: 0.0675 - val_acc: 0.9791
Epoch 25/100
Epoch 00024: val_loss did not improve
0s - loss: 0.2326 - acc: 0.5410 - val_loss: 0.0716 - val_acc: 0.9582
Epoch 26/100
Epoch 00025: val_loss improved from 0.06014 to 0.05747, saving model to static_7/weights.25-0.98.hdf5
0s - loss: 0.2381 - acc: 0.5149 - val_loss: 0.0575 - val_acc: 0.9803
Epoch 27/100
Epoch 00026: val_loss did not improve
0s - loss: 0.2326 - acc: 0.5337 - val_loss: 0.0637 - val_acc: 0.9816
Epoch 28/100
Epoch 00027: val_loss did not improve
0s - loss: 0.2359 - acc: 0.5211 - val_loss: 0.0620 - val_acc: 0.9828
Epoch 29/100
Epoch 00028: val_loss did not improve
0s - loss: 0.2345 - acc: 0.5284 - val_loss: 0.0699 - val_acc: 0.9681
Epoch 30/100
Epoch 00029: val_loss did not improve
0s - loss: 0.2333 - acc: 0.5238 - val_loss: 0.0650 - val_acc: 0.9742
Epoch 31/100
Epoch 00030: val_loss did not improve
0s - loss: 0.2325 - acc: 0.5340 - val_loss: 0.0646 - val_acc: 0.9742
Epoch 32/100
Epoch 00031: val_loss did not improve
0s - loss: 0.2365 - acc: 0.5223 - val_loss: 0.0693 - val_acc: 0.9730
Epoch 33/100
Epoch 00032: val_loss did not improve
0s - loss: 0.2323 - acc: 0.5284 - val_loss: 0.0756 - val_acc: 0.9693
Epoch 34/100
Epoch 00033: val_loss did not improve
0s - loss: 0.2304 - acc: 0.5355 - val_loss: 0.0695 - val_acc: 0.9668
Epoch 35/100
Epoch 00034: val_loss did not improve
0s - loss: 0.2348 - acc: 0.5272 - val_loss: 0.0641 - val_acc: 0.9779
Epoch 36/100
Epoch 00035: val_loss did not improve
0s - loss: 0.2332 - acc: 0.5278 - val_loss: 0.0664 - val_acc: 0.9791
Epoch 37/100
Epoch 00036: val_loss did not improve
0s - loss: 0.2342 - acc: 0.5247 - val_loss: 0.0641 - val_acc: 0.9767
Epoch 38/100
Epoch 00037: val_loss did not improve
0s - loss: 0.2316 - acc: 0.5370 - val_loss: 0.0623 - val_acc: 0.9779
Epoch 39/100
Epoch 00038: val_loss did not improve
0s - loss: 0.2318 - acc: 0.5327 - val_loss: 0.0678 - val_acc: 0.9730
Epoch 40/100
Epoch 00039: val_loss did not improve
0s - loss: 0.2282 - acc: 0.5398 - val_loss: 0.0675 - val_acc: 0.9828
Epoch 41/100
Epoch 00040: val_loss did not improve
0s - loss: 0.2372 - acc: 0.5217 - val_loss: 0.0634 - val_acc: 0.9681
Epoch 42/100
Epoch 00041: val_loss did not improve
0s - loss: 0.2331 - acc: 0.5321 - val_loss: 0.0761 - val_acc: 0.9398
Epoch 43/100
Epoch 00042: val_loss did not improve
0s - loss: 0.2321 - acc: 0.5324 - val_loss: 0.0705 - val_acc: 0.9705
Epoch 44/100
Epoch 00043: val_loss did not improve
0s - loss: 0.2312 - acc: 0.5377 - val_loss: 0.0636 - val_acc: 0.9730
Epoch 45/100
Epoch 00044: val_loss did not improve
0s - loss: 0.2333 - acc: 0.5324 - val_loss: 0.0599 - val_acc: 0.9816
Epoch 46/100
Epoch 00045: val_loss did not improve
0s - loss: 0.2315 - acc: 0.5330 - val_loss: 0.0634 - val_acc: 0.9754
Epoch 47/100
Epoch 00046: val_loss did not improve
0s - loss: 0.2317 - acc: 0.5349 - val_loss: 0.0691 - val_acc: 0.9767
Epoch 48/100
Epoch 00047: val_loss did not improve
0s - loss: 0.2248 - acc: 0.5515 - val_loss: 0.0700 - val_acc: 0.9779
Epoch 49/100
Epoch 00048: val_loss did not improve
0s - loss: 0.2327 - acc: 0.5340 - val_loss: 0.0612 - val_acc: 0.9803
Epoch 50/100
Epoch 00049: val_loss did not improve
0s - loss: 0.2327 - acc: 0.5263 - val_loss: 0.0639 - val_acc: 0.9803
Epoch 51/100
Epoch 00050: val_loss did not improve
0s - loss: 0.2348 - acc: 0.5238 - val_loss: 0.0673 - val_acc: 0.9754
Epoch 52/100
Epoch 00051: val_loss did not improve
0s - loss: 0.2336 - acc: 0.5272 - val_loss: 0.0674 - val_acc: 0.9779
Epoch 53/100
Epoch 00052: val_loss did not improve
0s - loss: 0.2263 - acc: 0.5509 - val_loss: 0.0724 - val_acc: 0.9816
Epoch 54/100
Epoch 00053: val_loss did not improve
0s - loss: 0.2350 - acc: 0.5241 - val_loss: 0.0616 - val_acc: 0.9840
Epoch 55/100
Epoch 00054: val_loss did not improve
0s - loss: 0.2338 - acc: 0.5232 - val_loss: 0.0620 - val_acc: 0.9803
Epoch 56/100
Epoch 00055: val_loss did not improve
0s - loss: 0.2268 - acc: 0.5444 - val_loss: 0.0699 - val_acc: 0.9644
Epoch 57/100
Epoch 00056: val_loss did not improve
0s - loss: 0.2322 - acc: 0.5294 - val_loss: 0.0708 - val_acc: 0.9730
Epoch 58/100
Epoch 00057: val_loss did not improve
0s - loss: 0.2284 - acc: 0.5401 - val_loss: 0.0665 - val_acc: 0.9717
Epoch 59/100
Epoch 00058: val_loss did not improve
0s - loss: 0.2332 - acc: 0.5281 - val_loss: 0.0580 - val_acc: 0.9853
Epoch 60/100
Epoch 00059: val_loss did not improve
0s - loss: 0.2288 - acc: 0.5386 - val_loss: 0.0643 - val_acc: 0.9853
Epoch 61/100
Epoch 00060: val_loss did not improve
0s - loss: 0.2289 - acc: 0.5401 - val_loss: 0.0595 - val_acc: 0.9914
Epoch 62/100
Epoch 00061: val_loss improved from 0.05747 to 0.05541, saving model to static_7/weights.61-0.99.hdf5
0s - loss: 0.2360 - acc: 0.5214 - val_loss: 0.0554 - val_acc: 0.9902
Epoch 63/100
Epoch 00062: val_loss did not improve
0s - loss: 0.2305 - acc: 0.5401 - val_loss: 0.0657 - val_acc: 0.9816
Epoch 64/100
Epoch 00063: val_loss did not improve
0s - loss: 0.2321 - acc: 0.5247 - val_loss: 0.0674 - val_acc: 0.9816
Epoch 65/100
Epoch 00064: val_loss did not improve
0s - loss: 0.2272 - acc: 0.5472 - val_loss: 0.0687 - val_acc: 0.9828
Epoch 66/100
Epoch 00065: val_loss did not improve
0s - loss: 0.2355 - acc: 0.5214 - val_loss: 0.0672 - val_acc: 0.9779
Epoch 67/100
Epoch 00066: val_loss did not improve
0s - loss: 0.2292 - acc: 0.5377 - val_loss: 0.0736 - val_acc: 0.9693
Epoch 68/100
Epoch 00067: val_loss improved from 0.05541 to 0.05453, saving model to static_7/weights.67-0.99.hdf5
0s - loss: 0.2304 - acc: 0.5349 - val_loss: 0.0545 - val_acc: 0.9877
Epoch 69/100
Epoch 00068: val_loss did not improve
0s - loss: 0.2296 - acc: 0.5346 - val_loss: 0.0639 - val_acc: 0.9828
Epoch 70/100
Epoch 00069: val_loss did not improve
0s - loss: 0.2283 - acc: 0.5398 - val_loss: 0.0655 - val_acc: 0.9779
Epoch 71/100
Epoch 00070: val_loss did not improve
0s - loss: 0.2317 - acc: 0.5324 - val_loss: 0.0660 - val_acc: 0.9791
Epoch 72/100
Epoch 00071: val_loss did not improve
0s - loss: 0.2362 - acc: 0.5152 - val_loss: 0.0631 - val_acc: 0.9853
Epoch 73/100
Epoch 00072: val_loss did not improve
0s - loss: 0.2309 - acc: 0.5321 - val_loss: 0.0641 - val_acc: 0.9853
Epoch 74/100
Epoch 00073: val_loss did not improve
0s - loss: 0.2345 - acc: 0.5204 - val_loss: 0.0631 - val_acc: 0.9877
Epoch 75/100
Epoch 00074: val_loss did not improve
0s - loss: 0.2294 - acc: 0.5380 - val_loss: 0.0641 - val_acc: 0.9853
Epoch 76/100
Epoch 00075: val_loss did not improve
0s - loss: 0.2290 - acc: 0.5450 - val_loss: 0.0662 - val_acc: 0.9803
Epoch 77/100
Epoch 00076: val_loss did not improve
0s - loss: 0.2326 - acc: 0.5272 - val_loss: 0.0623 - val_acc: 0.9853
Epoch 78/100
Epoch 00077: val_loss did not improve
0s - loss: 0.2266 - acc: 0.5417 - val_loss: 0.0682 - val_acc: 0.9693
Epoch 79/100
Epoch 00078: val_loss did not improve
0s - loss: 0.2299 - acc: 0.5361 - val_loss: 0.0591 - val_acc: 0.9865
Epoch 80/100
Epoch 00079: val_loss did not improve
0s - loss: 0.2332 - acc: 0.5294 - val_loss: 0.0622 - val_acc: 0.9853
Epoch 81/100
Epoch 00080: val_loss did not improve
0s - loss: 0.2282 - acc: 0.5423 - val_loss: 0.0700 - val_acc: 0.9717
Epoch 82/100
Epoch 00081: val_loss did not improve
0s - loss: 0.2273 - acc: 0.5401 - val_loss: 0.0703 - val_acc: 0.9816
Epoch 83/100
Epoch 00082: val_loss did not improve
0s - loss: 0.2328 - acc: 0.5204 - val_loss: 0.0616 - val_acc: 0.9865
Epoch 84/100
Epoch 00083: val_loss did not improve
0s - loss: 0.2297 - acc: 0.5346 - val_loss: 0.0619 - val_acc: 0.9865
Epoch 85/100
Epoch 00084: val_loss did not improve
0s - loss: 0.2314 - acc: 0.5266 - val_loss: 0.0644 - val_acc: 0.9828
Epoch 86/100
Epoch 00085: val_loss did not improve
0s - loss: 0.2313 - acc: 0.5324 - val_loss: 0.0699 - val_acc: 0.9754
Epoch 87/100
Epoch 00086: val_loss did not improve
0s - loss: 0.2330 - acc: 0.5235 - val_loss: 0.0642 - val_acc: 0.9853
Epoch 88/100
Epoch 00087: val_loss did not improve
0s - loss: 0.2315 - acc: 0.5327 - val_loss: 0.0608 - val_acc: 0.9865
Epoch 89/100
Epoch 00088: val_loss did not improve
0s - loss: 0.2258 - acc: 0.5457 - val_loss: 0.0608 - val_acc: 0.9865
Epoch 90/100
Epoch 00089: val_loss did not improve
0s - loss: 0.2336 - acc: 0.5208 - val_loss: 0.0563 - val_acc: 0.9865
Epoch 91/100
Epoch 00090: val_loss did not improve
0s - loss: 0.2304 - acc: 0.5297 - val_loss: 0.0650 - val_acc: 0.9840
Epoch 92/100
Epoch 00091: val_loss did not improve
0s - loss: 0.2272 - acc: 0.5398 - val_loss: 0.0643 - val_acc: 0.9840
Epoch 93/100
Epoch 00092: val_loss did not improve
0s - loss: 0.2332 - acc: 0.5192 - val_loss: 0.0589 - val_acc: 0.9853
Epoch 94/100
Epoch 00093: val_loss did not improve
0s - loss: 0.2254 - acc: 0.5518 - val_loss: 0.0636 - val_acc: 0.9779
Epoch 95/100
Epoch 00094: val_loss did not improve
0s - loss: 0.2314 - acc: 0.5287 - val_loss: 0.0606 - val_acc: 0.9865
Epoch 96/100
Epoch 00095: val_loss did not improve
0s - loss: 0.2284 - acc: 0.5370 - val_loss: 0.0698 - val_acc: 0.9767
Epoch 97/100
Epoch 00096: val_loss did not improve
0s - loss: 0.2277 - acc: 0.5453 - val_loss: 0.0634 - val_acc: 0.9853
Epoch 98/100
Epoch 00097: val_loss did not improve
0s - loss: 0.2249 - acc: 0.5444 - val_loss: 0.0617 - val_acc: 0.9840
Epoch 99/100
Epoch 00098: val_loss did not improve
0s - loss: 0.2308 - acc: 0.5324 - val_loss: 0.0649 - val_acc: 0.9828
Epoch 100/100
Epoch 00099: val_loss did not improve
0s - loss: 0.2347 - acc: 0.5217 - val_loss: 0.0633 - val_acc: 0.9840
(1560, 3)
------ STATIC ACCURACY ------
0.961538461538
[[439  52   0]
 [  7 525   0]
 [  1   0 536]]

Process finished with exit code 0


'''