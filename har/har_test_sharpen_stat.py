import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import random
import process_data

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code outputs the result of [UCI HAR (Static) : 1D CNN + Sharpen (%) : 96.67%]
given in Table 8. (Sensors 2018, 18(4), 1055, page 19 of 24)

The HAR model used is 'model/static.hdf5'.

'''

def display_output(title_str, X, y):

    # Load static HAR model
    model_path = "model/static.hdf5"
    model = load_model(model_path)

    print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
    print "===         Test Data: {} Half         ===\n".format(title_str)

    pred_dyna_raw = model.predict(np.expand_dims(X, axis=2), batch_size=32)
    print "---------------------------------"
    print "       NO SHARPEN ACCURACY       "
    print "---------------------------------"
    print accuracy_score(y, np.argmax(pred_dyna_raw, axis=1))
    print confusion_matrix(y, np.argmax(pred_dyna_raw, axis=1))

    print "\n---------------------------------"
    print "       SHARPENED  ACCURACY       "
    print "---------------------------------"

    alpha = np.arange(0.01, 0.31, 0.01)
    sigma = np.arange(5, 10, 1)

    for s in sigma:
        for a in alpha:
            # Sharpen test data with various sigma (for Gaussian filter) and alpha value combinations
            X_test_sharpen = process_data.sharpen(X, s, a)
            pred_dyna_sharpen = model.predict(np.expand_dims(X_test_sharpen, axis=2), batch_size=32)
            print ">>> sigma={}, alpha={:.2f}".format(s, a)
            print accuracy_score(y, np.argmax(pred_dyna_sharpen, axis=1))
            print confusion_matrix(y, np.argmax(pred_dyna_sharpen, axis=1))



# Load all test data (* dynamic and static data are mixed.)

X_test = process_data.load_x("test")
y_test = process_data.load_y("test")

# Set seed to ensure reproducibility of the paper.

seed = 818

# Static (4-sitting, 5-standing, 6-laying) test data are selected and
# split it in two, first & second, in order to determine
# sigma & alpha values for test data sharpening.

random.seed(seed)
stat_1 = np.where(y_test == 4)[0]
stat_1_first = random.sample(stat_1, int(len(stat_1) * 0.5))
stat_1_second = list(set(stat_1) - set(stat_1_first))

random.seed(seed)
stat_2 = np.where(y_test == 5)[0]
stat_2_first = random.sample(stat_2, int(len(stat_2) * 0.5))
stat_2_second = list(set(stat_2) - set(stat_2_first))

random.seed(seed)
stat_3 = np.where(y_test == 6)[0]
stat_3_first = random.sample(stat_3, int(len(stat_3) * 0.5))
stat_3_second = list(set(stat_3) - set(stat_3_first))

static_first = np.concatenate([stat_1_first, stat_2_first, stat_3_first])
static_second = np.concatenate([stat_1_second, stat_2_second, stat_3_second])

X_test_first = X_test[static_first]
y_test_first = y_test[static_first] - 4   # Convert (4, 5, 6) labels to (0, 1, 2)
print "test_static_first shape: ", X_test_first.shape

X_test_second = X_test[static_second]
y_test_second = y_test[static_second] - 4   # Convert (4, 5, 6) labels to (0, 1, 2)
print "test_static_second shape: ", X_test_second.shape


# Compare the static model accuracy:
# No sharpened (raw) test data vs. sharpened test data

display_output("First", X_test_first, y_test_first)
display_output("Second", X_test_second, y_test_second)