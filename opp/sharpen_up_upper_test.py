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
1D CNN UP position activity classification model using UPPER body TEST data.

The performance is measured using X_test, y_test dataset.

See right line graph in Figure 13 (Test Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''


X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("upper", "up")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===     [UPPER body sensors data] UP Class      ==="
print "===                1D CNN  MODEL                ==="
print "===          Evaluation on TEST DATA            ===\n"

# Load model
model = load_model('model/upper_up.hdf5')

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
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_up_upper_test.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===     [UPPER body sensors data] UP Class      ===
===                1D CNN  MODEL                ===
===          Evaluation on TEST DATA            ===

>>> RAW:
0.803821517751
[[5190  136]
 [1671 2214]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.822820540658
[[5128  198]
 [1434 2451]]
>>> SHARPENED: sigma=3, alpha=1.00
0.83020301813
[[5113  213]
 [1351 2534]]
>>> SHARPENED: sigma=3, alpha=1.50
0.832482900879
[[5097  229]
 [1314 2571]]
>>> SHARPENED: sigma=3, alpha=2.00
0.833242861796
[[5092  234]
 [1302 2583]]
>>> SHARPENED: sigma=3, alpha=2.50
0.834219954402
[[5088  238]
 [1289 2596]]
>>> SHARPENED: sigma=3, alpha=3.00
0.834762783628
[[5087  239]
 [1283 2602]]
>>> SHARPENED: sigma=3, alpha=3.50
0.835305612854
[[5087  239]
 [1278 2607]]
>>> SHARPENED: sigma=3, alpha=4.00
0.835522744545
[[5084  242]
 [1273 2612]]
>>> SHARPENED: sigma=3, alpha=4.50
0.835305612854
[[5083  243]
 [1274 2611]]
>>> SHARPENED: sigma=3, alpha=5.00
0.835739876235
[[5084  242]
 [1271 2614]]
>>> SHARPENED: sigma=3, alpha=5.50
0.835414178699
[[5081  245]
 [1271 2614]]
>>> SHARPENED: sigma=3, alpha=6.00
0.835522744545
[[5081  245]
 [1270 2615]]
>>> SHARPENED: sigma=3, alpha=6.50
0.83563131039
[[5080  246]
 [1268 2617]]
>>> SHARPENED: sigma=3, alpha=7.00
0.835739876235
[[5080  246]
 [1267 2618]]
>>> SHARPENED: sigma=3, alpha=7.50
0.835739876235
[[5080  246]
 [1267 2618]]
>>> SHARPENED: sigma=3, alpha=8.00
0.83563131039
[[5079  247]
 [1267 2618]]
>>> SHARPENED: sigma=3, alpha=8.50
0.835957007925
[[5079  247]
 [1264 2621]]
>>> SHARPENED: sigma=3, alpha=9.00
0.83606557377
[[5079  247]
 [1263 2622]]
>>> SHARPENED: sigma=3, alpha=9.50
0.836174139616
[[5079  247]
 [1262 2623]]
>>> SHARPENED: sigma=3, alpha=10.00
0.836391271306
[[5079  247]
 [1260 2625]]
>>> SHARPENED: sigma=3, alpha=10.50
0.836499837151
[[5079  247]
 [1259 2626]]
>>> SHARPENED: sigma=3, alpha=11.00
0.836499837151
[[5079  247]
 [1259 2626]]
>>> SHARPENED: sigma=3, alpha=11.50
0.836391271306
[[5078  248]
 [1259 2626]]
>>> SHARPENED: sigma=3, alpha=12.00
0.836608402996
[[5079  247]
 [1258 2627]]
>>> SHARPENED: sigma=3, alpha=12.50
0.836499837151
[[5079  247]
 [1259 2626]]
>>> SHARPENED: sigma=3, alpha=13.00
0.836825534687
[[5080  246]
 [1257 2628]]
>>> SHARPENED: sigma=3, alpha=13.50
0.836934100532
[[5080  246]
 [1256 2629]]
>>> SHARPENED: sigma=3, alpha=14.00
0.837042666377
[[5080  246]
 [1255 2630]]
>>> SHARPENED: sigma=3, alpha=14.50
0.837042666377
[[5080  246]
 [1255 2630]]
>>> SHARPENED: sigma=3, alpha=15.00
0.837042666377
[[5080  246]
 [1255 2630]]
>>> SHARPENED: sigma=4, alpha=0.50
0.821517750516
[[5126  200]
 [1444 2441]]
>>> SHARPENED: sigma=4, alpha=1.00
0.828683096298
[[5107  219]
 [1359 2526]]
>>> SHARPENED: sigma=4, alpha=1.50
0.831397242428
[[5092  234]
 [1319 2566]]
>>> SHARPENED: sigma=4, alpha=2.00
0.833242861796
[[5088  238]
 [1298 2587]]
>>> SHARPENED: sigma=4, alpha=2.50
0.83313429595
[[5083  243]
 [1294 2591]]
>>> SHARPENED: sigma=4, alpha=3.00
0.833351427641
[[5080  246]
 [1289 2596]]
>>> SHARPENED: sigma=4, alpha=3.50
0.833894256867
[[5077  249]
 [1281 2604]]
>>> SHARPENED: sigma=4, alpha=4.00
0.833785691022
[[5073  253]
 [1278 2607]]
>>> SHARPENED: sigma=4, alpha=4.50
0.833894256867
[[5071  255]
 [1275 2610]]
>>> SHARPENED: sigma=4, alpha=5.00
0.833894256867
[[5069  257]
 [1273 2612]]
>>> SHARPENED: sigma=4, alpha=5.50
0.834002822712
[[5069  257]
 [1272 2613]]
>>> SHARPENED: sigma=4, alpha=6.00
0.833894256867
[[5069  257]
 [1273 2612]]
>>> SHARPENED: sigma=4, alpha=6.50
0.833785691022
[[5068  258]
 [1273 2612]]
>>> SHARPENED: sigma=4, alpha=7.00
0.834111388557
[[5068  258]
 [1270 2615]]
>>> SHARPENED: sigma=4, alpha=7.50
0.834545651938
[[5069  257]
 [1267 2618]]
>>> SHARPENED: sigma=4, alpha=8.00
0.834328520248
[[5069  257]
 [1269 2616]]
>>> SHARPENED: sigma=4, alpha=8.50
0.834545651938
[[5069  257]
 [1267 2618]]
>>> SHARPENED: sigma=4, alpha=9.00
0.834545651938
[[5068  258]
 [1266 2619]]
>>> SHARPENED: sigma=4, alpha=9.50
0.834437086093
[[5068  258]
 [1267 2618]]
>>> SHARPENED: sigma=4, alpha=10.00
0.834654217783
[[5068  258]
 [1265 2620]]
>>> SHARPENED: sigma=4, alpha=10.50
0.834654217783
[[5068  258]
 [1265 2620]]
>>> SHARPENED: sigma=4, alpha=11.00
0.834654217783
[[5067  259]
 [1264 2621]]
>>> SHARPENED: sigma=4, alpha=11.50
0.834654217783
[[5066  260]
 [1263 2622]]
>>> SHARPENED: sigma=4, alpha=12.00
0.834654217783
[[5066  260]
 [1263 2622]]
>>> SHARPENED: sigma=4, alpha=12.50
0.834654217783
[[5066  260]
 [1263 2622]]
>>> SHARPENED: sigma=4, alpha=13.00
0.834762783628
[[5066  260]
 [1262 2623]]
>>> SHARPENED: sigma=4, alpha=13.50
0.834871349473
[[5067  259]
 [1262 2623]]
>>> SHARPENED: sigma=4, alpha=14.00
0.834871349473
[[5067  259]
 [1262 2623]]
>>> SHARPENED: sigma=4, alpha=14.50
0.834871349473
[[5067  259]
 [1262 2623]]
>>> SHARPENED: sigma=4, alpha=15.00
0.834979915319
[[5067  259]
 [1261 2624]]
>>> SHARPENED: sigma=5, alpha=0.50
0.820866355445
[[5122  204]
 [1446 2439]]
>>> SHARPENED: sigma=5, alpha=1.00
0.828031701227
[[5103  223]
 [1361 2524]]
>>> SHARPENED: sigma=5, alpha=1.50
0.830528715666
[[5087  239]
 [1322 2563]]
>>> SHARPENED: sigma=5, alpha=2.00
0.831722939963
[[5080  246]
 [1304 2581]]
>>> SHARPENED: sigma=5, alpha=2.50
0.831614374118
[[5072  254]
 [1297 2588]]
>>> SHARPENED: sigma=5, alpha=3.00
0.831397242428
[[5072  254]
 [1299 2586]]
>>> SHARPENED: sigma=5, alpha=3.50
0.831940071653
[[5069  257]
 [1291 2594]]
>>> SHARPENED: sigma=5, alpha=4.00
0.83270003257
[[5068  258]
 [1283 2602]]
>>> SHARPENED: sigma=5, alpha=4.50
0.833351427641
[[5068  258]
 [1277 2608]]
>>> SHARPENED: sigma=5, alpha=5.00
0.833351427641
[[5066  260]
 [1275 2610]]
>>> SHARPENED: sigma=5, alpha=5.50
0.833785691022
[[5066  260]
 [1271 2614]]
>>> SHARPENED: sigma=5, alpha=6.00
0.833785691022
[[5066  260]
 [1271 2614]]
>>> SHARPENED: sigma=5, alpha=6.50
0.833894256867
[[5066  260]
 [1270 2615]]
>>> SHARPENED: sigma=5, alpha=7.00
0.834002822712
[[5065  261]
 [1268 2617]]
>>> SHARPENED: sigma=5, alpha=7.50
0.833894256867
[[5064  262]
 [1268 2617]]
>>> SHARPENED: sigma=5, alpha=8.00
0.833785691022
[[5064  262]
 [1269 2616]]
>>> SHARPENED: sigma=5, alpha=8.50
0.833785691022
[[5062  264]
 [1267 2618]]
>>> SHARPENED: sigma=5, alpha=9.00
0.833785691022
[[5061  265]
 [1266 2619]]
>>> SHARPENED: sigma=5, alpha=9.50
0.833894256867
[[5061  265]
 [1265 2620]]
>>> SHARPENED: sigma=5, alpha=10.00
0.834219954402
[[5061  265]
 [1262 2623]]
>>> SHARPENED: sigma=5, alpha=10.50
0.834219954402
[[5060  266]
 [1261 2624]]
>>> SHARPENED: sigma=5, alpha=11.00
0.834219954402
[[5059  267]
 [1260 2625]]
>>> SHARPENED: sigma=5, alpha=11.50
0.834328520248
[[5060  266]
 [1260 2625]]
>>> SHARPENED: sigma=5, alpha=12.00
0.834328520248
[[5060  266]
 [1260 2625]]
>>> SHARPENED: sigma=5, alpha=12.50
0.834437086093
[[5060  266]
 [1259 2626]]
>>> SHARPENED: sigma=5, alpha=13.00
0.834545651938
[[5060  266]
 [1258 2627]]
>>> SHARPENED: sigma=5, alpha=13.50
0.834545651938
[[5060  266]
 [1258 2627]]
>>> SHARPENED: sigma=5, alpha=14.00
0.834437086093
[[5059  267]
 [1258 2627]]
>>> SHARPENED: sigma=5, alpha=14.50
0.834545651938
[[5059  267]
 [1257 2628]]
>>> SHARPENED: sigma=5, alpha=15.00
0.834654217783
[[5059  267]
 [1256 2629]]
>>> SHARPENED: sigma=6, alpha=0.50
0.820323526219
[[5122  204]
 [1451 2434]]
>>> SHARPENED: sigma=6, alpha=1.00
0.827380306156
[[5103  223]
 [1367 2518]]
>>> SHARPENED: sigma=6, alpha=1.50
0.830420149821
[[5090  236]
 [1326 2559]]
>>> SHARPENED: sigma=6, alpha=2.00
0.830854413202
[[5082  244]
 [1314 2571]]
>>> SHARPENED: sigma=6, alpha=2.50
0.831288676582
[[5077  249]
 [1305 2580]]
>>> SHARPENED: sigma=6, alpha=3.00
0.831722939963
[[5073  253]
 [1297 2588]]
>>> SHARPENED: sigma=6, alpha=3.50
0.831831505808
[[5071  255]
 [1294 2591]]
>>> SHARPENED: sigma=6, alpha=4.00
0.831831505808
[[5070  256]
 [1293 2592]]
>>> SHARPENED: sigma=6, alpha=4.50
0.832265769189
[[5070  256]
 [1289 2596]]
>>> SHARPENED: sigma=6, alpha=5.00
0.832808598415
[[5069  257]
 [1283 2602]]
>>> SHARPENED: sigma=6, alpha=5.50
0.832808598415
[[5069  257]
 [1283 2602]]
>>> SHARPENED: sigma=6, alpha=6.00
0.83270003257
[[5066  260]
 [1281 2604]]
>>> SHARPENED: sigma=6, alpha=6.50
0.83291716426
[[5066  260]
 [1279 2606]]
>>> SHARPENED: sigma=6, alpha=7.00
0.833242861796
[[5065  261]
 [1275 2610]]
>>> SHARPENED: sigma=6, alpha=7.50
0.833568559331
[[5065  261]
 [1272 2613]]
>>> SHARPENED: sigma=6, alpha=8.00
0.833351427641
[[5062  264]
 [1271 2614]]
>>> SHARPENED: sigma=6, alpha=8.50
0.83313429595
[[5060  266]
 [1271 2614]]
>>> SHARPENED: sigma=6, alpha=9.00
0.83313429595
[[5060  266]
 [1271 2614]]
>>> SHARPENED: sigma=6, alpha=9.50
0.833242861796
[[5060  266]
 [1270 2615]]
>>> SHARPENED: sigma=6, alpha=10.00
0.833242861796
[[5060  266]
 [1270 2615]]
>>> SHARPENED: sigma=6, alpha=10.50
0.833242861796
[[5060  266]
 [1270 2615]]
>>> SHARPENED: sigma=6, alpha=11.00
0.833242861796
[[5060  266]
 [1270 2615]]
>>> SHARPENED: sigma=6, alpha=11.50
0.83313429595
[[5059  267]
 [1270 2615]]
>>> SHARPENED: sigma=6, alpha=12.00
0.833025730105
[[5057  269]
 [1269 2616]]
>>> SHARPENED: sigma=6, alpha=12.50
0.833025730105
[[5058  268]
 [1270 2615]]
>>> SHARPENED: sigma=6, alpha=13.00
0.83313429595
[[5058  268]
 [1269 2616]]
>>> SHARPENED: sigma=6, alpha=13.50
0.833242861796
[[5058  268]
 [1268 2617]]
>>> SHARPENED: sigma=6, alpha=14.00
0.833242861796
[[5058  268]
 [1268 2617]]
>>> SHARPENED: sigma=6, alpha=14.50
0.83313429595
[[5057  269]
 [1268 2617]]
>>> SHARPENED: sigma=6, alpha=15.00
0.833025730105
[[5057  269]
 [1269 2616]]
>>> SHARPENED: sigma=7, alpha=0.50
0.819020736076
[[5124  202]
 [1465 2420]]
>>> SHARPENED: sigma=7, alpha=1.00
0.825643252633
[[5105  221]
 [1385 2500]]
>>> SHARPENED: sigma=7, alpha=1.50
0.828465964608
[[5095  231]
 [1349 2536]]
>>> SHARPENED: sigma=7, alpha=2.00
0.830420149821
[[5086  240]
 [1322 2563]]
>>> SHARPENED: sigma=7, alpha=2.50
0.831288676582
[[5084  242]
 [1312 2573]]
>>> SHARPENED: sigma=7, alpha=3.00
0.830854413202
[[5076  250]
 [1308 2577]]
>>> SHARPENED: sigma=7, alpha=3.50
0.831288676582
[[5072  254]
 [1300 2585]]
>>> SHARPENED: sigma=7, alpha=4.00
0.831722939963
[[5071  255]
 [1295 2590]]
>>> SHARPENED: sigma=7, alpha=4.50
0.831940071653
[[5071  255]
 [1293 2592]]
>>> SHARPENED: sigma=7, alpha=5.00
0.832265769189
[[5071  255]
 [1290 2595]]
>>> SHARPENED: sigma=7, alpha=5.50
0.832482900879
[[5071  255]
 [1288 2597]]
>>> SHARPENED: sigma=7, alpha=6.00
0.832808598415
[[5071  255]
 [1285 2600]]
>>> SHARPENED: sigma=7, alpha=6.50
0.832808598415
[[5070  256]
 [1284 2601]]
>>> SHARPENED: sigma=7, alpha=7.00
0.832808598415
[[5070  256]
 [1284 2601]]
>>> SHARPENED: sigma=7, alpha=7.50
0.832808598415
[[5070  256]
 [1284 2601]]
>>> SHARPENED: sigma=7, alpha=8.00
0.833025730105
[[5071  255]
 [1283 2602]]
>>> SHARPENED: sigma=7, alpha=8.50
0.833025730105
[[5068  258]
 [1280 2605]]
>>> SHARPENED: sigma=7, alpha=9.00
0.83270003257
[[5065  261]
 [1280 2605]]
>>> SHARPENED: sigma=7, alpha=9.50
0.832482900879
[[5064  262]
 [1281 2604]]
>>> SHARPENED: sigma=7, alpha=10.00
0.832591466725
[[5064  262]
 [1280 2605]]
>>> SHARPENED: sigma=7, alpha=10.50
0.832591466725
[[5064  262]
 [1280 2605]]
>>> SHARPENED: sigma=7, alpha=11.00
0.832808598415
[[5064  262]
 [1278 2607]]
>>> SHARPENED: sigma=7, alpha=11.50
0.83270003257
[[5063  263]
 [1278 2607]]
>>> SHARPENED: sigma=7, alpha=12.00
0.83270003257
[[5063  263]
 [1278 2607]]
>>> SHARPENED: sigma=7, alpha=12.50
0.83270003257
[[5063  263]
 [1278 2607]]
>>> SHARPENED: sigma=7, alpha=13.00
0.832808598415
[[5062  264]
 [1276 2609]]
>>> SHARPENED: sigma=7, alpha=13.50
0.832808598415
[[5062  264]
 [1276 2609]]
>>> SHARPENED: sigma=7, alpha=14.00
0.83291716426
[[5061  265]
 [1274 2611]]
>>> SHARPENED: sigma=7, alpha=14.50
0.832808598415
[[5060  266]
 [1274 2611]]
>>> SHARPENED: sigma=7, alpha=15.00
0.83270003257
[[5059  267]
 [1274 2611]]

Process finished with exit code 0
'''