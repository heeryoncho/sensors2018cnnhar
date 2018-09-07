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
1D CNN UP position activity classification model using LOWER body VALIDATION data.

The performance is measured using X_valid, y_valid dataset.

See left line graph in Figure 12 (Validation Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "up")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===     [LOWER body sensors data] UP Class      ==="
print "===                1D CNN  MODEL                ==="
print "===       Evaluation on VALIDATION DATA         ===\n"

# Load model
model = load_model('model/lower_up.hdf5')

print ">>> RAW:"
pred = model.predict(np.expand_dims(X_valid, axis=2), batch_size=32)
print accuracy_score(y_valid, np.argmax(pred, axis=1))
print confusion_matrix(y_valid, np.argmax(pred, axis=1)), '\n'

alpha = np.arange(0.5, 15.5, 0.5)
sigma = np.arange(3, 8, 1)

for s in sigma:
    for a in alpha:
        x_valid_sharpen = sd.sharpen(X_valid, s, a)
        pred_sharpened = model.predict(np.expand_dims(x_valid_sharpen, axis=2), batch_size=32)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_valid, np.argmax(pred_sharpened, axis=1))
        print confusion_matrix(y_valid, np.argmax(pred_sharpened, axis=1))


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_up_lower_valid.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===     [LOWER body sensors data] UP Class      ===
===                1D CNN  MODEL                ===
===       Evaluation on VALIDATION DATA         ===

>>> RAW:
0.857952069717
[[5159  805]
 [ 499 2717]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.864052287582
[[5213  751]
 [ 497 2719]]
>>> SHARPENED: sigma=3, alpha=1.00
0.866557734205
[[5226  738]
 [ 487 2729]]
>>> SHARPENED: sigma=3, alpha=1.50
0.866993464052
[[5225  739]
 [ 482 2734]]
>>> SHARPENED: sigma=3, alpha=2.00
0.8674291939
[[5231  733]
 [ 484 2732]]
>>> SHARPENED: sigma=3, alpha=2.50
0.86862745098
[[5238  726]
 [ 480 2736]]
>>> SHARPENED: sigma=3, alpha=3.00
0.869281045752
[[5242  722]
 [ 478 2738]]
>>> SHARPENED: sigma=3, alpha=3.50
0.869389978214
[[5241  723]
 [ 476 2740]]
>>> SHARPENED: sigma=3, alpha=4.00
0.869716775599
[[5244  720]
 [ 476 2740]]
>>> SHARPENED: sigma=3, alpha=4.50
0.869934640523
[[5247  717]
 [ 477 2739]]
>>> SHARPENED: sigma=3, alpha=5.00
0.870043572985
[[5250  714]
 [ 479 2737]]
>>> SHARPENED: sigma=3, alpha=5.50
0.869825708061
[[5248  716]
 [ 479 2737]]
>>> SHARPENED: sigma=3, alpha=6.00
0.869934640523
[[5248  716]
 [ 478 2738]]
>>> SHARPENED: sigma=3, alpha=6.50
0.87037037037
[[5249  715]
 [ 475 2741]]
>>> SHARPENED: sigma=3, alpha=7.00
0.87037037037
[[5249  715]
 [ 475 2741]]
>>> SHARPENED: sigma=3, alpha=7.50
0.87037037037
[[5249  715]
 [ 475 2741]]
>>> SHARPENED: sigma=3, alpha=8.00
0.870479302832
[[5249  715]
 [ 474 2742]]
>>> SHARPENED: sigma=3, alpha=8.50
0.870588235294
[[5250  714]
 [ 474 2742]]
>>> SHARPENED: sigma=3, alpha=9.00
0.870697167756
[[5251  713]
 [ 474 2742]]
>>> SHARPENED: sigma=3, alpha=9.50
0.87091503268
[[5253  711]
 [ 474 2742]]
>>> SHARPENED: sigma=3, alpha=10.00
0.871459694989
[[5256  708]
 [ 472 2744]]
>>> SHARPENED: sigma=3, alpha=10.50
0.872004357298
[[5258  706]
 [ 469 2747]]
>>> SHARPENED: sigma=3, alpha=11.00
0.87211328976
[[5258  706]
 [ 468 2748]]
>>> SHARPENED: sigma=3, alpha=11.50
0.87211328976
[[5258  706]
 [ 468 2748]]
>>> SHARPENED: sigma=3, alpha=12.00
0.87211328976
[[5258  706]
 [ 468 2748]]
>>> SHARPENED: sigma=3, alpha=12.50
0.872004357298
[[5258  706]
 [ 469 2747]]
>>> SHARPENED: sigma=3, alpha=13.00
0.87211328976
[[5259  705]
 [ 469 2747]]
>>> SHARPENED: sigma=3, alpha=13.50
0.871677559913
[[5256  708]
 [ 470 2746]]
>>> SHARPENED: sigma=3, alpha=14.00
0.871568627451
[[5256  708]
 [ 471 2745]]
>>> SHARPENED: sigma=3, alpha=14.50
0.871568627451
[[5256  708]
 [ 471 2745]]
>>> SHARPENED: sigma=3, alpha=15.00
0.871459694989
[[5255  709]
 [ 471 2745]]
>>> SHARPENED: sigma=4, alpha=0.50
0.862962962963
[[5201  763]
 [ 495 2721]]
>>> SHARPENED: sigma=4, alpha=1.00
0.864923747277
[[5213  751]
 [ 489 2727]]
>>> SHARPENED: sigma=4, alpha=1.50
0.866557734205
[[5220  744]
 [ 481 2735]]
>>> SHARPENED: sigma=4, alpha=2.00
0.86688453159
[[5221  743]
 [ 479 2737]]
>>> SHARPENED: sigma=4, alpha=2.50
0.867102396514
[[5223  741]
 [ 479 2737]]
>>> SHARPENED: sigma=4, alpha=3.00
0.867211328976
[[5224  740]
 [ 479 2737]]
>>> SHARPENED: sigma=4, alpha=3.50
0.867864923747
[[5229  735]
 [ 478 2738]]
>>> SHARPENED: sigma=4, alpha=4.00
0.868082788671
[[5231  733]
 [ 478 2738]]
>>> SHARPENED: sigma=4, alpha=4.50
0.86862745098
[[5235  729]
 [ 477 2739]]
>>> SHARPENED: sigma=4, alpha=5.00
0.869063180828
[[5238  726]
 [ 476 2740]]
>>> SHARPENED: sigma=4, alpha=5.50
0.869281045752
[[5240  724]
 [ 476 2740]]
>>> SHARPENED: sigma=4, alpha=6.00
0.869607843137
[[5243  721]
 [ 476 2740]]
>>> SHARPENED: sigma=4, alpha=6.50
0.869934640523
[[5245  719]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=7.00
0.869825708061
[[5244  720]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=7.50
0.869716775599
[[5243  721]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=8.00
0.869607843137
[[5243  721]
 [ 476 2740]]
>>> SHARPENED: sigma=4, alpha=8.50
0.869825708061
[[5243  721]
 [ 474 2742]]
>>> SHARPENED: sigma=4, alpha=9.00
0.869825708061
[[5243  721]
 [ 474 2742]]
>>> SHARPENED: sigma=4, alpha=9.50
0.869716775599
[[5243  721]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=10.00
0.869716775599
[[5243  721]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=10.50
0.869716775599
[[5243  721]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=11.00
0.869716775599
[[5243  721]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=11.50
0.869825708061
[[5244  720]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=12.00
0.869934640523
[[5245  719]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=12.50
0.869934640523
[[5245  719]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=13.00
0.869934640523
[[5245  719]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=13.50
0.870152505447
[[5247  717]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=14.00
0.870152505447
[[5247  717]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=14.50
0.870152505447
[[5247  717]
 [ 475 2741]]
>>> SHARPENED: sigma=4, alpha=15.00
0.870152505447
[[5248  716]
 [ 476 2740]]
>>> SHARPENED: sigma=5, alpha=0.50
0.86220043573
[[5197  767]
 [ 498 2718]]
>>> SHARPENED: sigma=5, alpha=1.00
0.864705882353
[[5213  751]
 [ 491 2725]]
>>> SHARPENED: sigma=5, alpha=1.50
0.865359477124
[[5216  748]
 [ 488 2728]]
>>> SHARPENED: sigma=5, alpha=2.00
0.866448801743
[[5224  740]
 [ 486 2730]]
>>> SHARPENED: sigma=5, alpha=2.50
0.8674291939
[[5232  732]
 [ 485 2731]]
>>> SHARPENED: sigma=5, alpha=3.00
0.867102396514
[[5230  734]
 [ 486 2730]]
>>> SHARPENED: sigma=5, alpha=3.50
0.867647058824
[[5232  732]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=4.00
0.867755991285
[[5233  731]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=4.50
0.868082788671
[[5236  728]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=5.00
0.868191721133
[[5237  727]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=5.50
0.868300653595
[[5237  727]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=6.00
0.868409586057
[[5238  726]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=6.50
0.868518518519
[[5240  724]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=7.00
0.868518518519
[[5241  723]
 [ 484 2732]]
>>> SHARPENED: sigma=5, alpha=7.50
0.86862745098
[[5242  722]
 [ 484 2732]]
>>> SHARPENED: sigma=5, alpha=8.00
0.868845315904
[[5243  721]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=8.50
0.86917211329
[[5246  718]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=9.00
0.869281045752
[[5247  717]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=9.50
0.869281045752
[[5247  717]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=10.00
0.869281045752
[[5247  717]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=10.50
0.869389978214
[[5248  716]
 [ 483 2733]]
>>> SHARPENED: sigma=5, alpha=11.00
0.869607843137
[[5248  716]
 [ 481 2735]]
>>> SHARPENED: sigma=5, alpha=11.50
0.869716775599
[[5250  714]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=12.00
0.869825708061
[[5251  713]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=12.50
0.869825708061
[[5251  713]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=13.00
0.870043572985
[[5252  712]
 [ 481 2735]]
>>> SHARPENED: sigma=5, alpha=13.50
0.869934640523
[[5252  712]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=14.00
0.869825708061
[[5251  713]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=14.50
0.869825708061
[[5251  713]
 [ 482 2734]]
>>> SHARPENED: sigma=5, alpha=15.00
0.869825708061
[[5251  713]
 [ 482 2734]]
>>> SHARPENED: sigma=6, alpha=0.50
0.861764705882
[[5197  767]
 [ 502 2714]]
>>> SHARPENED: sigma=6, alpha=1.00
0.864488017429
[[5218  746]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=1.50
0.8651416122
[[5221  743]
 [ 495 2721]]
>>> SHARPENED: sigma=6, alpha=2.00
0.865795206972
[[5226  738]
 [ 494 2722]]
>>> SHARPENED: sigma=6, alpha=2.50
0.866122004357
[[5230  734]
 [ 495 2721]]
>>> SHARPENED: sigma=6, alpha=3.00
0.866993464052
[[5238  726]
 [ 495 2721]]
>>> SHARPENED: sigma=6, alpha=3.50
0.866775599129
[[5240  724]
 [ 499 2717]]
>>> SHARPENED: sigma=6, alpha=4.00
0.866993464052
[[5241  723]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=4.50
0.867320261438
[[5246  718]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=5.00
0.8674291939
[[5246  718]
 [ 499 2717]]
>>> SHARPENED: sigma=6, alpha=5.50
0.867647058824
[[5248  716]
 [ 499 2717]]
>>> SHARPENED: sigma=6, alpha=6.00
0.868082788671
[[5250  714]
 [ 497 2719]]
>>> SHARPENED: sigma=6, alpha=6.50
0.868300653595
[[5253  711]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=7.00
0.868300653595
[[5253  711]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=7.50
0.868300653595
[[5253  711]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=8.00
0.868409586057
[[5254  710]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=8.50
0.86862745098
[[5256  708]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=9.00
0.868845315904
[[5258  706]
 [ 498 2718]]
>>> SHARPENED: sigma=6, alpha=9.50
0.86862745098
[[5258  706]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=10.00
0.86862745098
[[5259  705]
 [ 501 2715]]
>>> SHARPENED: sigma=6, alpha=10.50
0.868518518519
[[5259  705]
 [ 502 2714]]
>>> SHARPENED: sigma=6, alpha=11.00
0.868518518519
[[5259  705]
 [ 502 2714]]
>>> SHARPENED: sigma=6, alpha=11.50
0.86862745098
[[5260  704]
 [ 502 2714]]
>>> SHARPENED: sigma=6, alpha=12.00
0.868845315904
[[5261  703]
 [ 501 2715]]
>>> SHARPENED: sigma=6, alpha=12.50
0.868954248366
[[5261  703]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=13.00
0.868954248366
[[5261  703]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=13.50
0.869063180828
[[5262  702]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=14.00
0.869063180828
[[5262  702]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=14.50
0.869063180828
[[5262  702]
 [ 500 2716]]
>>> SHARPENED: sigma=6, alpha=15.00
0.869281045752
[[5264  700]
 [ 500 2716]]
>>> SHARPENED: sigma=7, alpha=0.50
0.86220043573
[[5203  761]
 [ 504 2712]]
>>> SHARPENED: sigma=7, alpha=1.00
0.864705882353
[[5225  739]
 [ 503 2713]]
>>> SHARPENED: sigma=7, alpha=1.50
0.865250544662
[[5227  737]
 [ 500 2716]]
>>> SHARPENED: sigma=7, alpha=2.00
0.866013071895
[[5234  730]
 [ 500 2716]]
>>> SHARPENED: sigma=7, alpha=2.50
0.866339869281
[[5238  726]
 [ 501 2715]]
>>> SHARPENED: sigma=7, alpha=3.00
0.866993464052
[[5246  718]
 [ 503 2713]]
>>> SHARPENED: sigma=7, alpha=3.50
0.867102396514
[[5248  716]
 [ 504 2712]]
>>> SHARPENED: sigma=7, alpha=4.00
0.867211328976
[[5250  714]
 [ 505 2711]]
>>> SHARPENED: sigma=7, alpha=4.50
0.8674291939
[[5252  712]
 [ 505 2711]]
>>> SHARPENED: sigma=7, alpha=5.00
0.867973856209
[[5257  707]
 [ 505 2711]]
>>> SHARPENED: sigma=7, alpha=5.50
0.868082788671
[[5258  706]
 [ 505 2711]]
>>> SHARPENED: sigma=7, alpha=6.00
0.868300653595
[[5261  703]
 [ 506 2710]]
>>> SHARPENED: sigma=7, alpha=6.50
0.868191721133
[[5262  702]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=7.00
0.868082788671
[[5262  702]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=7.50
0.868300653595
[[5264  700]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=8.00
0.868300653595
[[5264  700]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=8.50
0.868300653595
[[5264  700]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=9.00
0.868300653595
[[5264  700]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=9.50
0.868409586057
[[5265  699]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=10.00
0.868518518519
[[5266  698]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=10.50
0.86862745098
[[5266  698]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=11.00
0.868736383442
[[5267  697]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=11.50
0.868518518519
[[5266  698]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=12.00
0.868518518519
[[5266  698]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=12.50
0.868518518519
[[5266  698]
 [ 509 2707]]
>>> SHARPENED: sigma=7, alpha=13.00
0.86862745098
[[5266  698]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=13.50
0.86862745098
[[5266  698]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=14.00
0.86862745098
[[5266  698]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=14.50
0.868736383442
[[5267  697]
 [ 508 2708]]
>>> SHARPENED: sigma=7, alpha=15.00
0.868845315904
[[5267  697]
 [ 507 2709]]

Process finished with exit code 0
'''