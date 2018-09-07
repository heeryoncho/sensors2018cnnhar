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
1D CNN UP position activity classification model using UPPER body VALIDATION data.

The performance is measured using X_valid, y_valid dataset.

See left line graph in Figure 13 (Validation Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''


X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("upper", "up")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===     [UPPER body sensors data] UP Class      ==="
print "===                1D CNN  MODEL                ==="
print "===       Evaluation on VALIDATION DATA         ===\n"

# Load model
model = load_model('model/upper_up.hdf5')

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
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_up_upper_valid.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===     [UPPER body sensors data] UP Class      ===
===                1D CNN  MODEL                ===
===       Evaluation on VALIDATION DATA         ===

>>> RAW:
0.793790849673
[[5421  543]
 [1350 1866]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.79651416122
[[5269  695]
 [1173 2043]]
>>> SHARPENED: sigma=3, alpha=1.00
0.795969498911
[[5200  764]
 [1109 2107]]
>>> SHARPENED: sigma=3, alpha=1.50
0.795751633987
[[5159  805]
 [1070 2146]]
>>> SHARPENED: sigma=3, alpha=2.00
0.797058823529
[[5149  815]
 [1048 2168]]
>>> SHARPENED: sigma=3, alpha=2.50
0.796840958606
[[5134  830]
 [1035 2181]]
>>> SHARPENED: sigma=3, alpha=3.00
0.796623093682
[[5123  841]
 [1026 2190]]
>>> SHARPENED: sigma=3, alpha=3.50
0.796296296296
[[5112  852]
 [1018 2198]]
>>> SHARPENED: sigma=3, alpha=4.00
0.795860566449
[[5106  858]
 [1016 2200]]
>>> SHARPENED: sigma=3, alpha=4.50
0.796078431373
[[5106  858]
 [1014 2202]]
>>> SHARPENED: sigma=3, alpha=5.00
0.796296296296
[[5107  857]
 [1013 2203]]
>>> SHARPENED: sigma=3, alpha=5.50
0.796078431373
[[5105  859]
 [1013 2203]]
>>> SHARPENED: sigma=3, alpha=6.00
0.796078431373
[[5102  862]
 [1010 2206]]
>>> SHARPENED: sigma=3, alpha=6.50
0.796078431373
[[5100  864]
 [1008 2208]]
>>> SHARPENED: sigma=3, alpha=7.00
0.796078431373
[[5099  865]
 [1007 2209]]
>>> SHARPENED: sigma=3, alpha=7.50
0.796405228758
[[5098  866]
 [1003 2213]]
>>> SHARPENED: sigma=3, alpha=8.00
0.796732026144
[[5098  866]
 [1000 2216]]
>>> SHARPENED: sigma=3, alpha=8.50
0.796840958606
[[5099  865]
 [1000 2216]]
>>> SHARPENED: sigma=3, alpha=9.00
0.796840958606
[[5098  866]
 [ 999 2217]]
>>> SHARPENED: sigma=3, alpha=9.50
0.796840958606
[[5098  866]
 [ 999 2217]]
>>> SHARPENED: sigma=3, alpha=10.00
0.797058823529
[[5099  865]
 [ 998 2218]]
>>> SHARPENED: sigma=3, alpha=10.50
0.797167755991
[[5100  864]
 [ 998 2218]]
>>> SHARPENED: sigma=3, alpha=11.00
0.797385620915
[[5100  864]
 [ 996 2220]]
>>> SHARPENED: sigma=3, alpha=11.50
0.797385620915
[[5100  864]
 [ 996 2220]]
>>> SHARPENED: sigma=3, alpha=12.00
0.797385620915
[[5099  865]
 [ 995 2221]]
>>> SHARPENED: sigma=3, alpha=12.50
0.797385620915
[[5099  865]
 [ 995 2221]]
>>> SHARPENED: sigma=3, alpha=13.00
0.797276688453
[[5098  866]
 [ 995 2221]]
>>> SHARPENED: sigma=3, alpha=13.50
0.796840958606
[[5094  870]
 [ 995 2221]]
>>> SHARPENED: sigma=3, alpha=14.00
0.796732026144
[[5094  870]
 [ 996 2220]]
>>> SHARPENED: sigma=3, alpha=14.50
0.796732026144
[[5094  870]
 [ 996 2220]]
>>> SHARPENED: sigma=3, alpha=15.00
0.796405228758
[[5091  873]
 [ 996 2220]]
>>> SHARPENED: sigma=4, alpha=0.50
0.796078431373
[[5268  696]
 [1176 2040]]
>>> SHARPENED: sigma=4, alpha=1.00
0.795315904139
[[5200  764]
 [1115 2101]]
>>> SHARPENED: sigma=4, alpha=1.50
0.793790849673
[[5153  811]
 [1082 2134]]
>>> SHARPENED: sigma=4, alpha=2.00
0.794335511983
[[5134  830]
 [1058 2158]]
>>> SHARPENED: sigma=4, alpha=2.50
0.794335511983
[[5117  847]
 [1041 2175]]
>>> SHARPENED: sigma=4, alpha=3.00
0.794335511983
[[5110  854]
 [1034 2182]]
>>> SHARPENED: sigma=4, alpha=3.50
0.794117647059
[[5102  862]
 [1028 2188]]
>>> SHARPENED: sigma=4, alpha=4.00
0.794008714597
[[5096  868]
 [1023 2193]]
>>> SHARPENED: sigma=4, alpha=4.50
0.793899782135
[[5089  875]
 [1017 2199]]
>>> SHARPENED: sigma=4, alpha=5.00
0.793464052288
[[5084  880]
 [1016 2200]]
>>> SHARPENED: sigma=4, alpha=5.50
0.793572984749
[[5081  883]
 [1012 2204]]
>>> SHARPENED: sigma=4, alpha=6.00
0.793464052288
[[5078  886]
 [1010 2206]]
>>> SHARPENED: sigma=4, alpha=6.50
0.793790849673
[[5077  887]
 [1006 2210]]
>>> SHARPENED: sigma=4, alpha=7.00
0.793464052288
[[5073  891]
 [1005 2211]]
>>> SHARPENED: sigma=4, alpha=7.50
0.793355119826
[[5071  893]
 [1004 2212]]
>>> SHARPENED: sigma=4, alpha=8.00
0.793246187364
[[5069  895]
 [1003 2213]]
>>> SHARPENED: sigma=4, alpha=8.50
0.793246187364
[[5068  896]
 [1002 2214]]
>>> SHARPENED: sigma=4, alpha=9.00
0.793137254902
[[5067  897]
 [1002 2214]]
>>> SHARPENED: sigma=4, alpha=9.50
0.792919389978
[[5065  899]
 [1002 2214]]
>>> SHARPENED: sigma=4, alpha=10.00
0.792701525054
[[5062  902]
 [1001 2215]]
>>> SHARPENED: sigma=4, alpha=10.50
0.792592592593
[[5060  904]
 [1000 2216]]
>>> SHARPENED: sigma=4, alpha=11.00
0.792483660131
[[5058  906]
 [ 999 2217]]
>>> SHARPENED: sigma=4, alpha=11.50
0.792374727669
[[5057  907]
 [ 999 2217]]
>>> SHARPENED: sigma=4, alpha=12.00
0.792265795207
[[5055  909]
 [ 998 2218]]
>>> SHARPENED: sigma=4, alpha=12.50
0.792265795207
[[5054  910]
 [ 997 2219]]
>>> SHARPENED: sigma=4, alpha=13.00
0.792483660131
[[5054  910]
 [ 995 2221]]
>>> SHARPENED: sigma=4, alpha=13.50
0.792483660131
[[5054  910]
 [ 995 2221]]
>>> SHARPENED: sigma=4, alpha=14.00
0.792374727669
[[5053  911]
 [ 995 2221]]
>>> SHARPENED: sigma=4, alpha=14.50
0.792047930283
[[5051  913]
 [ 996 2220]]
>>> SHARPENED: sigma=4, alpha=15.00
0.791938997821
[[5050  914]
 [ 996 2220]]
>>> SHARPENED: sigma=5, alpha=0.50
0.796078431373
[[5272  692]
 [1180 2036]]
>>> SHARPENED: sigma=5, alpha=1.00
0.79477124183
[[5203  761]
 [1123 2093]]
>>> SHARPENED: sigma=5, alpha=1.50
0.793899782135
[[5165  799]
 [1093 2123]]
>>> SHARPENED: sigma=5, alpha=2.00
0.792265795207
[[5133  831]
 [1076 2140]]
>>> SHARPENED: sigma=5, alpha=2.50
0.793464052288
[[5120  844]
 [1052 2164]]
>>> SHARPENED: sigma=5, alpha=3.00
0.793572984749
[[5104  860]
 [1035 2181]]
>>> SHARPENED: sigma=5, alpha=3.50
0.793572984749
[[5099  865]
 [1030 2186]]
>>> SHARPENED: sigma=5, alpha=4.00
0.793464052288
[[5094  870]
 [1026 2190]]
>>> SHARPENED: sigma=5, alpha=4.50
0.79302832244
[[5087  877]
 [1023 2193]]
>>> SHARPENED: sigma=5, alpha=5.00
0.792919389978
[[5084  880]
 [1021 2195]]
>>> SHARPENED: sigma=5, alpha=5.50
0.792919389978
[[5081  883]
 [1018 2198]]
>>> SHARPENED: sigma=5, alpha=6.00
0.793246187364
[[5081  883]
 [1015 2201]]
>>> SHARPENED: sigma=5, alpha=6.50
0.793464052288
[[5081  883]
 [1013 2203]]
>>> SHARPENED: sigma=5, alpha=7.00
0.793572984749
[[5079  885]
 [1010 2206]]
>>> SHARPENED: sigma=5, alpha=7.50
0.793137254902
[[5075  889]
 [1010 2206]]
>>> SHARPENED: sigma=5, alpha=8.00
0.793137254902
[[5074  890]
 [1009 2207]]
>>> SHARPENED: sigma=5, alpha=8.50
0.79302832244
[[5072  892]
 [1008 2208]]
>>> SHARPENED: sigma=5, alpha=9.00
0.792810457516
[[5069  895]
 [1007 2209]]
>>> SHARPENED: sigma=5, alpha=9.50
0.792592592593
[[5066  898]
 [1006 2210]]
>>> SHARPENED: sigma=5, alpha=10.00
0.792592592593
[[5066  898]
 [1006 2210]]
>>> SHARPENED: sigma=5, alpha=10.50
0.792701525054
[[5066  898]
 [1005 2211]]
>>> SHARPENED: sigma=5, alpha=11.00
0.792919389978
[[5066  898]
 [1003 2213]]
>>> SHARPENED: sigma=5, alpha=11.50
0.79302832244
[[5066  898]
 [1002 2214]]
>>> SHARPENED: sigma=5, alpha=12.00
0.792919389978
[[5065  899]
 [1002 2214]]
>>> SHARPENED: sigma=5, alpha=12.50
0.793137254902
[[5064  900]
 [ 999 2217]]
>>> SHARPENED: sigma=5, alpha=13.00
0.792810457516
[[5062  902]
 [1000 2216]]
>>> SHARPENED: sigma=5, alpha=13.50
0.792701525054
[[5062  902]
 [1001 2215]]
>>> SHARPENED: sigma=5, alpha=14.00
0.792592592593
[[5061  903]
 [1001 2215]]
>>> SHARPENED: sigma=5, alpha=14.50
0.792483660131
[[5060  904]
 [1001 2215]]
>>> SHARPENED: sigma=5, alpha=15.00
0.792592592593
[[5060  904]
 [1000 2216]]
>>> SHARPENED: sigma=6, alpha=0.50
0.794335511983
[[5278  686]
 [1202 2014]]
>>> SHARPENED: sigma=6, alpha=1.00
0.792592592593
[[5205  759]
 [1145 2071]]
>>> SHARPENED: sigma=6, alpha=1.50
0.792810457516
[[5176  788]
 [1114 2102]]
>>> SHARPENED: sigma=6, alpha=2.00
0.791503267974
[[5150  814]
 [1100 2116]]
>>> SHARPENED: sigma=6, alpha=2.50
0.792156862745
[[5132  832]
 [1076 2140]]
>>> SHARPENED: sigma=6, alpha=3.00
0.792919389978
[[5119  845]
 [1056 2160]]
>>> SHARPENED: sigma=6, alpha=3.50
0.792265795207
[[5109  855]
 [1052 2164]]
>>> SHARPENED: sigma=6, alpha=4.00
0.792156862745
[[5105  859]
 [1049 2167]]
>>> SHARPENED: sigma=6, alpha=4.50
0.792047930283
[[5101  863]
 [1046 2170]]
>>> SHARPENED: sigma=6, alpha=5.00
0.792047930283
[[5097  867]
 [1042 2174]]
>>> SHARPENED: sigma=6, alpha=5.50
0.792265795207
[[5095  869]
 [1038 2178]]
>>> SHARPENED: sigma=6, alpha=6.00
0.791721132898
[[5090  874]
 [1038 2178]]
>>> SHARPENED: sigma=6, alpha=6.50
0.791394335512
[[5086  878]
 [1037 2179]]
>>> SHARPENED: sigma=6, alpha=7.00
0.791503267974
[[5085  879]
 [1035 2181]]
>>> SHARPENED: sigma=6, alpha=7.50
0.791612200436
[[5085  879]
 [1034 2182]]
>>> SHARPENED: sigma=6, alpha=8.00
0.791612200436
[[5085  879]
 [1034 2182]]
>>> SHARPENED: sigma=6, alpha=8.50
0.791503267974
[[5084  880]
 [1034 2182]]
>>> SHARPENED: sigma=6, alpha=9.00
0.79128540305
[[5082  882]
 [1034 2182]]
>>> SHARPENED: sigma=6, alpha=9.50
0.791394335512
[[5082  882]
 [1033 2183]]
>>> SHARPENED: sigma=6, alpha=10.00
0.791503267974
[[5082  882]
 [1032 2184]]
>>> SHARPENED: sigma=6, alpha=10.50
0.79128540305
[[5079  885]
 [1031 2185]]
>>> SHARPENED: sigma=6, alpha=11.00
0.791394335512
[[5079  885]
 [1030 2186]]
>>> SHARPENED: sigma=6, alpha=11.50
0.791394335512
[[5079  885]
 [1030 2186]]
>>> SHARPENED: sigma=6, alpha=12.00
0.791394335512
[[5079  885]
 [1030 2186]]
>>> SHARPENED: sigma=6, alpha=12.50
0.791503267974
[[5079  885]
 [1029 2187]]
>>> SHARPENED: sigma=6, alpha=13.00
0.791503267974
[[5079  885]
 [1029 2187]]
>>> SHARPENED: sigma=6, alpha=13.50
0.791503267974
[[5079  885]
 [1029 2187]]
>>> SHARPENED: sigma=6, alpha=14.00
0.791503267974
[[5078  886]
 [1028 2188]]
>>> SHARPENED: sigma=6, alpha=14.50
0.791394335512
[[5077  887]
 [1028 2188]]
>>> SHARPENED: sigma=6, alpha=15.00
0.791394335512
[[5077  887]
 [1028 2188]]
>>> SHARPENED: sigma=7, alpha=0.50
0.793790849673
[[5283  681]
 [1212 2004]]
>>> SHARPENED: sigma=7, alpha=1.00
0.792156862745
[[5210  754]
 [1154 2062]]
>>> SHARPENED: sigma=7, alpha=1.50
0.793464052288
[[5189  775]
 [1121 2095]]
>>> SHARPENED: sigma=7, alpha=2.00
0.792483660131
[[5165  799]
 [1106 2110]]
>>> SHARPENED: sigma=7, alpha=2.50
0.792810457516
[[5152  812]
 [1090 2126]]
>>> SHARPENED: sigma=7, alpha=3.00
0.793572984749
[[5142  822]
 [1073 2143]]
>>> SHARPENED: sigma=7, alpha=3.50
0.793464052288
[[5133  831]
 [1065 2151]]
>>> SHARPENED: sigma=7, alpha=4.00
0.793355119826
[[5125  839]
 [1058 2158]]
>>> SHARPENED: sigma=7, alpha=4.50
0.793355119826
[[5121  843]
 [1054 2162]]
>>> SHARPENED: sigma=7, alpha=5.00
0.793355119826
[[5119  845]
 [1052 2164]]
>>> SHARPENED: sigma=7, alpha=5.50
0.793572984749
[[5117  847]
 [1048 2168]]
>>> SHARPENED: sigma=7, alpha=6.00
0.793137254902
[[5112  852]
 [1047 2169]]
>>> SHARPENED: sigma=7, alpha=6.50
0.793137254902
[[5110  854]
 [1045 2171]]
>>> SHARPENED: sigma=7, alpha=7.00
0.792919389978
[[5106  858]
 [1043 2173]]
>>> SHARPENED: sigma=7, alpha=7.50
0.79302832244
[[5106  858]
 [1042 2174]]
>>> SHARPENED: sigma=7, alpha=8.00
0.792701525054
[[5103  861]
 [1042 2174]]
>>> SHARPENED: sigma=7, alpha=8.50
0.792919389978
[[5104  860]
 [1041 2175]]
>>> SHARPENED: sigma=7, alpha=9.00
0.792810457516
[[5101  863]
 [1039 2177]]
>>> SHARPENED: sigma=7, alpha=9.50
0.792919389978
[[5101  863]
 [1038 2178]]
>>> SHARPENED: sigma=7, alpha=10.00
0.792919389978
[[5101  863]
 [1038 2178]]
>>> SHARPENED: sigma=7, alpha=10.50
0.79302832244
[[5101  863]
 [1037 2179]]
>>> SHARPENED: sigma=7, alpha=11.00
0.79302832244
[[5101  863]
 [1037 2179]]
>>> SHARPENED: sigma=7, alpha=11.50
0.79302832244
[[5100  864]
 [1036 2180]]
>>> SHARPENED: sigma=7, alpha=12.00
0.792919389978
[[5099  865]
 [1036 2180]]
>>> SHARPENED: sigma=7, alpha=12.50
0.792919389978
[[5099  865]
 [1036 2180]]
>>> SHARPENED: sigma=7, alpha=13.00
0.792810457516
[[5098  866]
 [1036 2180]]
>>> SHARPENED: sigma=7, alpha=13.50
0.792701525054
[[5097  867]
 [1036 2180]]
>>> SHARPENED: sigma=7, alpha=14.00
0.792810457516
[[5096  868]
 [1034 2182]]
>>> SHARPENED: sigma=7, alpha=14.50
0.792592592593
[[5094  870]
 [1034 2182]]
>>> SHARPENED: sigma=7, alpha=15.00
0.792592592593
[[5094  870]
 [1034 2182]]

Process finished with exit code 0
'''