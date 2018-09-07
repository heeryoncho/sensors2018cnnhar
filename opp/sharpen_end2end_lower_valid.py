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
1D CNN End-to-End activity classification model using LOWER body VALIDATION data.

The performance is measured using X_valid, y_valid dataset.

See left line graph in Figure 14 (a) (Validation Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "end2end")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===   [LOWER body sensors data] End2End Class   ==="
print "===                1D CNN  MODEL                ==="
print "===       Evaluation on VALIDATION DATA         ===\n"

# Load model
model = load_model('model/lower_end2end.hdf5')

print ">>> RAW:"
pred = model.predict(np.expand_dims(X_valid, axis=2), batch_size=32)
print accuracy_score(y_valid, np.argmax(pred, axis=1))
print confusion_matrix(y_valid, np.argmax(pred, axis=1)), '\n'

alpha = np.arange(0.5, 3.5, 0.5)
sigma = np.arange(3, 8, 1)

for s in sigma:
    for a in alpha:
        x_valid_sharpen = sd.sharpen(X_valid, s, a)
        pred_sharpened = model.predict(np.expand_dims(x_valid_sharpen, axis=2), batch_size=32)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_valid, np.argmax(pred_sharpened, axis=1))
        print confusion_matrix(y_valid, np.argmax(pred_sharpened, axis=1))


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_end2end_lower_valid.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===   [LOWER body sensors data] End2End Class   ===
===                1D CNN  MODEL                ===
===       Evaluation on VALIDATION DATA         ===

>>> RAW:
0.881255051804
[[4936  608  420    0]
 [ 555 2658    3    0]
 [  27    3 3736    0]
 [   0    0    0  663]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.878903666691
[[5024  529  411    0]
 [ 619 2588    9    0]
 [  80    0 3686    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=3, alpha=1.00
0.8759644353
[[5059  498  407    0]
 [ 663 2538   15    0]
 [ 105    0 3661    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=3, alpha=1.50
0.872657799985
[[5063  497  404    0]
 [ 694 2504   18    0]
 [ 120    0 3646    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=3, alpha=2.00
0.871996472922
[[5092  474  398    0]
 [ 718 2479   19    0]
 [ 133    0 3633    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=3, alpha=2.50
0.870894261151
[[5105  464  395    0]
 [ 739 2456   21    0]
 [ 138    0 3628    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=3, alpha=3.00
0.869277683886
[[5115  454  395    0]
 [ 762 2431   23    0]
 [ 145    0 3621    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=4, alpha=0.50
0.878903666691
[[5029  527  408    0]
 [ 624 2583    9    0]
 [  80    0 3686    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=4, alpha=1.00
0.876037916085
[[5054  504  406    0]
 [ 657 2548   11    0]
 [ 109    0 3657    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=4, alpha=1.50
0.874347858035
[[5079  486  399    0]
 [ 686 2516   14    0]
 [ 125    0 3641    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=4, alpha=2.00
0.873025203909
[[5095  473  396    0]
 [ 709 2492   15    0]
 [ 135    0 3631    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=4, alpha=2.50
0.872069953707
[[5109  463  392    0]
 [ 730 2471   15    0]
 [ 141    0 3625    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=4, alpha=3.00
0.871408626644
[[5117  455  392    0]
 [ 742 2458   16    0]
 [ 145    0 3621    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=5, alpha=0.50
0.87919758983
[[5036  522  406    0]
 [ 627 2582    7    0]
 [  82    0 3684    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=5, alpha=1.00
0.876258358439
[[5062  499  403    0]
 [ 660 2546   10    0]
 [ 112    0 3654    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=5, alpha=1.50
0.874347858035
[[5088  479  397    0]
 [ 693 2511   12    0]
 [ 129    0 3637    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=5, alpha=2.00
0.873613050187
[[5104  466  394    0]
 [ 707 2495   14    0]
 [ 139    0 3627    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=5, alpha=2.50
0.873098684694
[[5119  455  390    0]
 [ 724 2477   15    0]
 [ 143    0 3623    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=5, alpha=3.00
0.871996472922
[[5127  447  390    0]
 [ 739 2462   15    0]
 [ 151    0 3615    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=6, alpha=0.50
0.879564993754
[[5043  516  405    0]
 [ 628 2581    7    0]
 [  83    0 3683    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=6, alpha=1.00
0.87611139687
[[5066  495  403    0]
 [ 664 2542   10    0]
 [ 114    0 3652    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=6, alpha=1.50
0.874641781174
[[5098  472  394    0]
 [ 698 2507   11    0]
 [ 131    0 3635    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=6, alpha=2.00
0.873392607833
[[5109  463  392    0]
 [ 714 2488   14    0]
 [ 140    0 3626    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=6, alpha=2.50
0.871996472922
[[5117  457  390    0]
 [ 733 2469   14    0]
 [ 148    0 3618    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=6, alpha=3.00
0.871555588214
[[5126  448  390    0]
 [ 743 2458   15    0]
 [ 152    0 3614    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=7, alpha=0.50
0.879638474539
[[5048  514  402    0]
 [ 629 2580    7    0]
 [  86    0 3680    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=7, alpha=1.00
0.876184877654
[[5067  495  402    0]
 [ 664 2543    9    0]
 [ 115    0 3651    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=7, alpha=1.50
0.874788742744
[[5101  468  395    0]
 [ 698 2507   11    0]
 [ 132    0 3634    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=7, alpha=2.00
0.873539569403
[[5117  455  392    0]
 [ 719 2483   14    0]
 [ 141    0 3625    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=7, alpha=2.50
0.872657799985
[[5126  448  390    0]
 [ 733 2469   14    0]
 [ 148    0 3618    0]
 [   0    0    0  663]]
>>> SHARPENED: sigma=7, alpha=3.00
0.871922992138
[[5130  444  390    0]
 [ 741 2460   15    0]
 [ 153    0 3613    0]
 [   0    0    0  663]]

Process finished with exit code 0
'''