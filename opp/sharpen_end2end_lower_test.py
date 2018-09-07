import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model
import select_data as sd

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code investigates the effects of test data sharpening on 
1D CNN End-to-End activity classification model using LOWER body TEST data.

The performance is measured using X_test, y_test dataset.

See right line graph in Figure 14 (a) (Test Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "end2end")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===   [LOWER body sensors data] End2End Class   ==="
print "===                1D CNN  MODEL                ==="
print "===           Evaluation on TEST DATA           ===\n"

# Load model
model = load_model('model/lower_end2end.hdf5')

print ">>> RAW:"
pred = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
print accuracy_score(y_test, np.argmax(pred, axis=1))
print confusion_matrix(y_test, np.argmax(pred, axis=1)), '\n'

alpha = np.arange(0.5, 3.5, 0.5)
sigma = np.arange(3, 8, 1)

for s in sigma:
    for a in alpha:
        x_test_sharpen = sd.sharpen(X_test, s, a)
        pred_sharpened = model.predict(np.expand_dims(x_test_sharpen, axis=2), batch_size=32)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_test, np.argmax(pred_sharpened, axis=1))
        print confusion_matrix(y_test, np.argmax(pred_sharpened, axis=1))


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_end2end_lower_test.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===   [LOWER body sensors data] End2End Class   ===
===                1D CNN  MODEL                ===
===           Evaluation on TEST DATA           ===

>>> RAW:
0.932709447415
[[5039  253   32    2]
 [ 589 3280   16    0]
 [  14    0 3446    0]
 [   0    0    0  793]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.931818181818
[[5112  187   27    0]
 [ 669 3201   15    0]
 [  20    0 3440    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=3, alpha=1.00
0.927361853832
[[5150  155   21    0]
 [ 735 3128   22    0]
 [  45    0 3415    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=3, alpha=1.50
0.915701128936
[[5169  140   17    0]
 [ 792 3070   23    0]
 [ 163    0 3297    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=3, alpha=2.00
0.890002970885
[[5177  133   16    0]
 [ 821 3038   25    1]
 [ 485    0 2975    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=3, alpha=2.50
0.815953654189
[[5182  132   12    0]
 [ 857 3002   25    1]
 [1451    0 2009    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=3, alpha=3.00
0.731209150327
[[5193  125    8    0]
 [ 884 2976   25    0]
 [2577    0  883    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=4, alpha=0.50
0.932263814617
[[5111  191   24    0]
 [ 663 3208   14    0]
 [  20    0 3440    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=4, alpha=1.00
0.92803030303
[[5152  156   18    0]
 [ 728 3138   19    0]
 [  48    0 3412    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=4, alpha=1.50
0.918226381462
[[5162  148   16    0]
 [ 769 3094   22    0]
 [ 146    0 3314    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=4, alpha=2.00
0.894533571004
[[5174  139   13    0]
 [ 805 3060   20    0]
 [ 443    0 3017    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=4, alpha=2.50
0.838012477718
[[5185  132    9    0]
 [ 836 3028   21    0]
 [1183    0 2277    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=4, alpha=3.00
0.758318478907
[[5191  128    7    0]
 [ 853 3011   21    0]
 [2245    0 1215    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=5, alpha=0.50
0.93233808675
[[5111  193   22    0]
 [ 662 3209   14    0]
 [  20    0 3440    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=5, alpha=1.00
0.928773024361
[[5152  156   18    0]
 [ 720 3149   16    0]
 [  49    0 3411    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=5, alpha=1.50
0.919414735591
[[5164  146   16    0]
 [ 761 3103   21    0]
 [ 141    0 3319    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=5, alpha=2.00
0.896167557932
[[5178  138   10    0]
 [ 795 3067   23    0]
 [ 432    0 3028    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=5, alpha=2.50
0.844251336898
[[5187  132    7    0]
 [ 817 3044   24    0]
 [1117    0 2343    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=5, alpha=3.00
0.766191325015
[[5193  127    6    0]
 [ 846 3018   21    0]
 [2148    0 1312    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=6, alpha=0.50
0.932412358883
[[5115  190   21    0]
 [ 665 3206   14    0]
 [  20    0 3440    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=6, alpha=1.00
0.92825311943
[[5148  161   17    0]
 [ 722 3147   16    0]
 [  50    0 3410    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=6, alpha=1.50
0.919414735591
[[5166  145   15    0]
 [ 761 3105   19    0]
 [ 145    0 3315    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=6, alpha=2.00
0.897281639929
[[5179  137   10    0]
 [ 790 3073   22    0]
 [ 424    0 3036    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=6, alpha=2.50
0.848633392751
[[5191  129    6    0]
 [ 817 3043   25    0]
 [1061    0 2399    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=6, alpha=3.00
0.770796197267
[[5195  125    6    0]
 [ 845 3018   22    0]
 [2088    0 1372    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=7, alpha=0.50
0.932189542484
[[5115  190   21    0]
 [ 669 3202   14    0]
 [  19    0 3441    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=7, alpha=1.00
0.928327391563
[[5147  162   17    0]
 [ 722 3149   14    0]
 [  50    0 3410    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=7, alpha=1.50
0.919711824124
[[5168  143   15    0]
 [ 765 3102   18    0]
 [ 140    0 3320    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=7, alpha=2.00
0.898618538324
[[5179  137   10    0]
 [ 793 3071   21    0]
 [ 404    0 3056    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=7, alpha=2.50
0.85197563874
[[5190  130    6    0]
 [ 825 3037   23    0]
 [1009    0 2451    0]
 [   0    0    0  793]]
>>> SHARPENED: sigma=7, alpha=3.00
0.778000594177
[[5197  123    6    0]
 [ 844 3016   25    0]
 [1991    0 1469    0]
 [   0    0    0  793]]

Process finished with exit code 0
'''