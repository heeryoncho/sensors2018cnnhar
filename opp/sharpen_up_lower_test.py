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
1D CNN UP position activity classification model using LOWER body TEST data.

The performance is measured using X_test, y_test dataset.

See right line graph in Figure 12 (Test Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''


X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "up")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===     [LOWER body sensors data] UP Class      ==="
print "===                1D CNN  MODEL                ==="
print "===          Evaluation on TEST DATA            ===\n"

# Load model
model = load_model('model/lower_up.hdf5')

print ">>> RAW:"
pred = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
print accuracy_score(y_test, np.argmax(pred, axis=1))
print confusion_matrix(y_test, np.argmax(pred, axis=1)), '\n'

alpha = np.arange(0.5, 15.5, 0.5)
sigma = np.arange(3, 8, 1)

for s in sigma:
    for a in alpha:
        x_test_sharpen = sd.sharpen(X_test, s, a)
        pred_sharpened = model.predict(np.expand_dims(x_test_sharpen, axis=2), batch_size=32)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_test, np.argmax(pred_sharpened, axis=1))
        print confusion_matrix(y_test, np.argmax(pred_sharpened, axis=1))


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_up_lower_test.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===     [LOWER body sensors data] UP Class      ===
===                1D CNN  MODEL                ===
===          Evaluation on TEST DATA            ===

>>> RAW:
0.912821626316
[[5183  143]
 [ 660 3225]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.915101509065
[[5204  122]
 [ 660 3225]]
>>> SHARPENED: sigma=3, alpha=1.00
0.915752904136
[[5212  114]
 [ 662 3223]]
>>> SHARPENED: sigma=3, alpha=1.50
0.915318640756
[[5215  111]
 [ 669 3216]]
>>> SHARPENED: sigma=3, alpha=2.00
0.915970035827
[[5218  108]
 [ 666 3219]]
>>> SHARPENED: sigma=3, alpha=2.50
0.916295733362
[[5219  107]
 [ 664 3221]]
>>> SHARPENED: sigma=3, alpha=3.00
0.916295733362
[[5219  107]
 [ 664 3221]]
>>> SHARPENED: sigma=3, alpha=3.50
0.916078601672
[[5217  109]
 [ 664 3221]]
>>> SHARPENED: sigma=3, alpha=4.00
0.916295733362
[[5217  109]
 [ 662 3223]]
>>> SHARPENED: sigma=3, alpha=4.50
0.916404299207
[[5216  110]
 [ 660 3225]]
>>> SHARPENED: sigma=3, alpha=5.00
0.916838562588
[[5217  109]
 [ 657 3228]]
>>> SHARPENED: sigma=3, alpha=5.50
0.916838562588
[[5216  110]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=6.00
0.916729996743
[[5215  111]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=6.50
0.916838562588
[[5215  111]
 [ 655 3230]]
>>> SHARPENED: sigma=3, alpha=7.00
0.916404299207
[[5213  113]
 [ 657 3228]]
>>> SHARPENED: sigma=3, alpha=7.50
0.916295733362
[[5213  113]
 [ 658 3227]]
>>> SHARPENED: sigma=3, alpha=8.00
0.916187167517
[[5213  113]
 [ 659 3226]]
>>> SHARPENED: sigma=3, alpha=8.50
0.916404299207
[[5212  114]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=9.00
0.916512865053
[[5213  113]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=9.50
0.916404299207
[[5212  114]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=10.00
0.916404299207
[[5212  114]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=10.50
0.916295733362
[[5211  115]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=11.00
0.916295733362
[[5211  115]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=11.50
0.916295733362
[[5210  116]
 [ 655 3230]]
>>> SHARPENED: sigma=3, alpha=12.00
0.916295733362
[[5210  116]
 [ 655 3230]]
>>> SHARPENED: sigma=3, alpha=12.50
0.916187167517
[[5210  116]
 [ 656 3229]]
>>> SHARPENED: sigma=3, alpha=13.00
0.916295733362
[[5210  116]
 [ 655 3230]]
>>> SHARPENED: sigma=3, alpha=13.50
0.916295733362
[[5210  116]
 [ 655 3230]]
>>> SHARPENED: sigma=3, alpha=14.00
0.916404299207
[[5211  115]
 [ 655 3230]]
>>> SHARPENED: sigma=3, alpha=14.50
0.916621430898
[[5211  115]
 [ 653 3232]]
>>> SHARPENED: sigma=3, alpha=15.00
0.916621430898
[[5211  115]
 [ 653 3232]]
>>> SHARPENED: sigma=4, alpha=0.50
0.915101509065
[[5202  124]
 [ 658 3227]]
>>> SHARPENED: sigma=4, alpha=1.00
0.915535772446
[[5208  118]
 [ 660 3225]]
>>> SHARPENED: sigma=4, alpha=1.50
0.915535772446
[[5215  111]
 [ 667 3218]]
>>> SHARPENED: sigma=4, alpha=2.00
0.915318640756
[[5216  110]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=2.50
0.915427206601
[[5217  109]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=3.00
0.915752904136
[[5219  107]
 [ 669 3216]]
>>> SHARPENED: sigma=4, alpha=3.50
0.915861469982
[[5219  107]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=4.00
0.915970035827
[[5220  106]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=4.50
0.915752904136
[[5218  108]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=5.00
0.915318640756
[[5217  109]
 [ 671 3214]]
>>> SHARPENED: sigma=4, alpha=5.50
0.915535772446
[[5218  108]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=6.00
0.915644338291
[[5218  108]
 [ 669 3216]]
>>> SHARPENED: sigma=4, alpha=6.50
0.915427206601
[[5218  108]
 [ 671 3214]]
>>> SHARPENED: sigma=4, alpha=7.00
0.915427206601
[[5218  108]
 [ 671 3214]]
>>> SHARPENED: sigma=4, alpha=7.50
0.915427206601
[[5218  108]
 [ 671 3214]]
>>> SHARPENED: sigma=4, alpha=8.00
0.915535772446
[[5218  108]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=8.50
0.915535772446
[[5218  108]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=9.00
0.915427206601
[[5217  109]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=9.50
0.915427206601
[[5217  109]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=10.00
0.915318640756
[[5216  110]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=10.50
0.915318640756
[[5216  110]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=11.00
0.915427206601
[[5217  109]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=11.50
0.915427206601
[[5217  109]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=12.00
0.915427206601
[[5217  109]
 [ 670 3215]]
>>> SHARPENED: sigma=4, alpha=12.50
0.915644338291
[[5217  109]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=13.00
0.915644338291
[[5217  109]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=13.50
0.915644338291
[[5217  109]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=14.00
0.915535772446
[[5216  110]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=14.50
0.915535772446
[[5216  110]
 [ 668 3217]]
>>> SHARPENED: sigma=4, alpha=15.00
0.915427206601
[[5215  111]
 [ 668 3217]]
>>> SHARPENED: sigma=5, alpha=0.50
0.914558679839
[[5201  125]
 [ 662 3223]]
>>> SHARPENED: sigma=5, alpha=1.00
0.91521007491
[[5210  116]
 [ 665 3220]]
>>> SHARPENED: sigma=5, alpha=1.50
0.91499294322
[[5216  110]
 [ 673 3212]]
>>> SHARPENED: sigma=5, alpha=2.00
0.914884377375
[[5219  107]
 [ 677 3208]]
>>> SHARPENED: sigma=5, alpha=2.50
0.915427206601
[[5221  105]
 [ 674 3211]]
>>> SHARPENED: sigma=5, alpha=3.00
0.915427206601
[[5222  104]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=3.50
0.915535772446
[[5224  102]
 [ 676 3209]]
>>> SHARPENED: sigma=5, alpha=4.00
0.915752904136
[[5225  101]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=4.50
0.915970035827
[[5225  101]
 [ 673 3212]]
>>> SHARPENED: sigma=5, alpha=5.00
0.916078601672
[[5226  100]
 [ 673 3212]]
>>> SHARPENED: sigma=5, alpha=5.50
0.915970035827
[[5226  100]
 [ 674 3211]]
>>> SHARPENED: sigma=5, alpha=6.00
0.915970035827
[[5226  100]
 [ 674 3211]]
>>> SHARPENED: sigma=5, alpha=6.50
0.915970035827
[[5225  101]
 [ 673 3212]]
>>> SHARPENED: sigma=5, alpha=7.00
0.915970035827
[[5225  101]
 [ 673 3212]]
>>> SHARPENED: sigma=5, alpha=7.50
0.915644338291
[[5225  101]
 [ 676 3209]]
>>> SHARPENED: sigma=5, alpha=8.00
0.915318640756
[[5224  102]
 [ 678 3207]]
>>> SHARPENED: sigma=5, alpha=8.50
0.915318640756
[[5224  102]
 [ 678 3207]]
>>> SHARPENED: sigma=5, alpha=9.00
0.915318640756
[[5224  102]
 [ 678 3207]]
>>> SHARPENED: sigma=5, alpha=9.50
0.915644338291
[[5225  101]
 [ 676 3209]]
>>> SHARPENED: sigma=5, alpha=10.00
0.915644338291
[[5225  101]
 [ 676 3209]]
>>> SHARPENED: sigma=5, alpha=10.50
0.915644338291
[[5225  101]
 [ 676 3209]]
>>> SHARPENED: sigma=5, alpha=11.00
0.915752904136
[[5226  100]
 [ 676 3209]]
>>> SHARPENED: sigma=5, alpha=11.50
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=12.00
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=12.50
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=13.00
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=13.50
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=14.00
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=14.50
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=5, alpha=15.00
0.915861469982
[[5226  100]
 [ 675 3210]]
>>> SHARPENED: sigma=6, alpha=0.50
0.914341548149
[[5203  123]
 [ 666 3219]]
>>> SHARPENED: sigma=6, alpha=1.00
0.91477581153
[[5213  113]
 [ 672 3213]]
>>> SHARPENED: sigma=6, alpha=1.50
0.91477581153
[[5219  107]
 [ 678 3207]]
>>> SHARPENED: sigma=6, alpha=2.00
0.914884377375
[[5221  105]
 [ 679 3206]]
>>> SHARPENED: sigma=6, alpha=2.50
0.91477581153
[[5225  101]
 [ 684 3201]]
>>> SHARPENED: sigma=6, alpha=3.00
0.914667245685
[[5225  101]
 [ 685 3200]]
>>> SHARPENED: sigma=6, alpha=3.50
0.91477581153
[[5227   99]
 [ 686 3199]]
>>> SHARPENED: sigma=6, alpha=4.00
0.914884377375
[[5227   99]
 [ 685 3200]]
>>> SHARPENED: sigma=6, alpha=4.50
0.91477581153
[[5227   99]
 [ 686 3199]]
>>> SHARPENED: sigma=6, alpha=5.00
0.914667245685
[[5227   99]
 [ 687 3198]]
>>> SHARPENED: sigma=6, alpha=5.50
0.914884377375
[[5228   98]
 [ 686 3199]]
>>> SHARPENED: sigma=6, alpha=6.00
0.91477581153
[[5230   96]
 [ 689 3196]]
>>> SHARPENED: sigma=6, alpha=6.50
0.91477581153
[[5231   95]
 [ 690 3195]]
>>> SHARPENED: sigma=6, alpha=7.00
0.91499294322
[[5232   94]
 [ 689 3196]]
>>> SHARPENED: sigma=6, alpha=7.50
0.91499294322
[[5232   94]
 [ 689 3196]]
>>> SHARPENED: sigma=6, alpha=8.00
0.91499294322
[[5232   94]
 [ 689 3196]]
>>> SHARPENED: sigma=6, alpha=8.50
0.914884377375
[[5232   94]
 [ 690 3195]]
>>> SHARPENED: sigma=6, alpha=9.00
0.914884377375
[[5232   94]
 [ 690 3195]]
>>> SHARPENED: sigma=6, alpha=9.50
0.91477581153
[[5232   94]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=10.00
0.91477581153
[[5232   94]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=10.50
0.91477581153
[[5232   94]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=11.00
0.914884377375
[[5233   93]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=11.50
0.914884377375
[[5233   93]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=12.00
0.914884377375
[[5233   93]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=12.50
0.91477581153
[[5232   94]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=13.00
0.91477581153
[[5232   94]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=13.50
0.91477581153
[[5232   94]
 [ 691 3194]]
>>> SHARPENED: sigma=6, alpha=14.00
0.914558679839
[[5232   94]
 [ 693 3192]]
>>> SHARPENED: sigma=6, alpha=14.50
0.914558679839
[[5232   94]
 [ 693 3192]]
>>> SHARPENED: sigma=6, alpha=15.00
0.914450113994
[[5231   95]
 [ 693 3192]]
>>> SHARPENED: sigma=7, alpha=0.50
0.914124416459
[[5204  122]
 [ 669 3216]]
>>> SHARPENED: sigma=7, alpha=1.00
0.914450113994
[[5217  109]
 [ 679 3206]]
>>> SHARPENED: sigma=7, alpha=1.50
0.914450113994
[[5222  104]
 [ 684 3201]]
>>> SHARPENED: sigma=7, alpha=2.00
0.914558679839
[[5225  101]
 [ 686 3199]]
>>> SHARPENED: sigma=7, alpha=2.50
0.914341548149
[[5226  100]
 [ 689 3196]]
>>> SHARPENED: sigma=7, alpha=3.00
0.91499294322
[[5229   97]
 [ 686 3199]]
>>> SHARPENED: sigma=7, alpha=3.50
0.914884377375
[[5229   97]
 [ 687 3198]]
>>> SHARPENED: sigma=7, alpha=4.00
0.91477581153
[[5229   97]
 [ 688 3197]]
>>> SHARPENED: sigma=7, alpha=4.50
0.914667245685
[[5230   96]
 [ 690 3195]]
>>> SHARPENED: sigma=7, alpha=5.00
0.914450113994
[[5230   96]
 [ 692 3193]]
>>> SHARPENED: sigma=7, alpha=5.50
0.914667245685
[[5232   94]
 [ 692 3193]]
>>> SHARPENED: sigma=7, alpha=6.00
0.914667245685
[[5232   94]
 [ 692 3193]]
>>> SHARPENED: sigma=7, alpha=6.50
0.914558679839
[[5232   94]
 [ 693 3192]]
>>> SHARPENED: sigma=7, alpha=7.00
0.914558679839
[[5233   93]
 [ 694 3191]]
>>> SHARPENED: sigma=7, alpha=7.50
0.914450113994
[[5234   92]
 [ 696 3189]]
>>> SHARPENED: sigma=7, alpha=8.00
0.914450113994
[[5235   91]
 [ 697 3188]]
>>> SHARPENED: sigma=7, alpha=8.50
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=9.00
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=9.50
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=10.00
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=10.50
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=11.00
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=11.50
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=12.00
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=12.50
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=13.00
0.914450113994
[[5236   90]
 [ 698 3187]]
>>> SHARPENED: sigma=7, alpha=13.50
0.914341548149
[[5236   90]
 [ 699 3186]]
>>> SHARPENED: sigma=7, alpha=14.00
0.914450113994
[[5237   89]
 [ 699 3186]]
>>> SHARPENED: sigma=7, alpha=14.50
0.914450113994
[[5237   89]
 [ 699 3186]]
>>> SHARPENED: sigma=7, alpha=15.00
0.914450113994
[[5237   89]
 [ 699 3186]]

Process finished with exit code 0

'''
