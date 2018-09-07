import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import select_data as sd


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code investigates the effects of test data sharpening on baseline
ML techniques, Logistic Regression & Random Forest.

The performance is measured using LOWER body X_valid, y_valid dataset.

See left line graphs in Figure 14 (b) & (c) (Validation Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "up")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===     [LOWER body sensors data]  UP Class     ==="
print "===     Logistic Regression & Random Forest     ==="
print "===        Evaluation on VALIDATION DATA        ===\n"

print "\n============================================"
print "          LOGISTIC REGRESSION"
print "============================================\n"

clf_lr = LogisticRegression(random_state=2018)
clf_lr.fit(X_train, y_train)

print ">>> RAW:"
pred_lr = clf_lr.predict(X_valid)
print accuracy_score(y_valid, pred_lr)
print confusion_matrix(y_valid, pred_lr), '\n'

alpha = np.arange(0.5, 3.5, 0.5)
sigma = np.arange(3, 8, 1)

for s in sigma:
    for a in alpha:
        x_valid_sharpen = sd.sharpen(X_valid, s, a)
        pred_lr = clf_lr.predict(x_valid_sharpen)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_valid, pred_lr)
        print confusion_matrix(y_valid, pred_lr)


print "\n============================================"
print "               RANDOM FOREST"
print "============================================\n"

clf_rf = RandomForestClassifier(random_state=2018, max_depth=5, n_estimators=10, max_features=1)
clf_rf.fit(X_train, y_train)

print ">>> RAW:"
pred_rf = clf_rf.predict(X_valid)
print accuracy_score(y_valid, pred_rf)
print confusion_matrix(y_valid, pred_rf), '\n'

for s in sigma:
    for a in alpha:
        x_valid_sharpen = sd.sharpen(X_valid, s, a)
        pred_rf = clf_rf.predict(x_valid_sharpen)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_valid, pred_rf)
        print confusion_matrix(y_valid, pred_rf)



'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_baseline_lrrf_lower_valid.py

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===     [LOWER body sensors data]  UP Class     ===
===     Logistic Regression & Random Forest     ===
===        Evaluation on VALIDATION DATA        ===


============================================
          LOGISTIC REGRESSION
============================================

>>> RAW:
0.833551198257
[[5189  775]
 [ 753 2463]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.819063180828
[[4661 1303]
 [ 358 2858]]
>>> SHARPENED: sigma=3, alpha=1.00
0.810130718954
[[4610 1354]
 [ 389 2827]]
>>> SHARPENED: sigma=3, alpha=1.50
0.805991285403
[[4584 1380]
 [ 401 2815]]
>>> SHARPENED: sigma=3, alpha=2.00
0.804139433551
[[4570 1394]
 [ 404 2812]]
>>> SHARPENED: sigma=3, alpha=2.50
0.802178649237
[[4557 1407]
 [ 409 2807]]
>>> SHARPENED: sigma=3, alpha=3.00
0.800762527233
[[4549 1415]
 [ 414 2802]]
>>> SHARPENED: sigma=4, alpha=0.50
0.822875816993
[[4713 1251]
 [ 375 2841]]
>>> SHARPENED: sigma=4, alpha=1.00
0.812854030501
[[4651 1313]
 [ 405 2811]]
>>> SHARPENED: sigma=4, alpha=1.50
0.80871459695
[[4624 1340]
 [ 416 2800]]
>>> SHARPENED: sigma=4, alpha=2.00
0.806862745098
[[4609 1355]
 [ 418 2798]]
>>> SHARPENED: sigma=4, alpha=2.50
0.805010893246
[[4596 1368]
 [ 422 2794]]
>>> SHARPENED: sigma=4, alpha=3.00
0.804030501089
[[4590 1374]
 [ 425 2791]]
>>> SHARPENED: sigma=5, alpha=0.50
0.82614379085
[[4756 1208]
 [ 388 2828]]
>>> SHARPENED: sigma=5, alpha=1.00
0.816013071895
[[4686 1278]
 [ 411 2805]]
>>> SHARPENED: sigma=5, alpha=1.50
0.812527233115
[[4674 1290]
 [ 431 2785]]
>>> SHARPENED: sigma=5, alpha=2.00
0.810130718954
[[4656 1308]
 [ 435 2781]]
>>> SHARPENED: sigma=5, alpha=2.50
0.808823529412
[[4644 1320]
 [ 435 2781]]
>>> SHARPENED: sigma=5, alpha=3.00
0.807734204793
[[4637 1327]
 [ 438 2778]]
>>> SHARPENED: sigma=6, alpha=0.50
0.828649237473
[[4796 1168]
 [ 405 2811]]
>>> SHARPENED: sigma=6, alpha=1.00
0.817211328976
[[4708 1256]
 [ 422 2794]]
>>> SHARPENED: sigma=6, alpha=1.50
0.813725490196
[[4691 1273]
 [ 437 2779]]
>>> SHARPENED: sigma=6, alpha=2.00
0.810021786492
[[4668 1296]
 [ 448 2768]]
>>> SHARPENED: sigma=6, alpha=2.50
0.807734204793
[[4654 1310]
 [ 455 2761]]
>>> SHARPENED: sigma=6, alpha=3.00
0.806209150327
[[4644 1320]
 [ 459 2757]]
>>> SHARPENED: sigma=7, alpha=0.50
0.830174291939
[[4821 1143]
 [ 416 2800]]
>>> SHARPENED: sigma=7, alpha=1.00
0.81862745098
[[4735 1229]
 [ 436 2780]]
>>> SHARPENED: sigma=7, alpha=1.50
0.812854030501
[[4695 1269]
 [ 449 2767]]
>>> SHARPENED: sigma=7, alpha=2.00
0.809368191721
[[4672 1292]
 [ 458 2758]]
>>> SHARPENED: sigma=7, alpha=2.50
0.807516339869
[[4663 1301]
 [ 466 2750]]
>>> SHARPENED: sigma=7, alpha=3.00
0.806862745098
[[4661 1303]
 [ 470 2746]]

============================================
               RANDOM FOREST
============================================

>>> RAW:
0.838126361656
[[5283  681]
 [ 805 2411]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.823638344227
[[4849 1115]
 [ 504 2712]]
>>> SHARPENED: sigma=3, alpha=1.00
0.782788671024
[[4244 1720]
 [ 274 2942]]
>>> SHARPENED: sigma=3, alpha=1.50
0.744226579521
[[3788 2176]
 [ 172 3044]]
>>> SHARPENED: sigma=3, alpha=2.00
0.708169934641
[[3420 2544]
 [ 135 3081]]
>>> SHARPENED: sigma=3, alpha=2.50
0.68660130719
[[3217 2747]
 [ 130 3086]]
>>> SHARPENED: sigma=3, alpha=3.00
0.661328976035
[[2982 2982]
 [ 127 3089]]
>>> SHARPENED: sigma=4, alpha=0.50
0.817973856209
[[4749 1215]
 [ 456 2760]]
>>> SHARPENED: sigma=4, alpha=1.00
0.774291938998
[[4152 1812]
 [ 260 2956]]
>>> SHARPENED: sigma=4, alpha=1.50
0.738126361656
[[3690 2274]
 [ 130 3086]]
>>> SHARPENED: sigma=4, alpha=2.00
0.700544662309
[[3317 2647]
 [ 102 3114]]
>>> SHARPENED: sigma=4, alpha=2.50
0.6825708061
[[3138 2826]
 [  88 3128]]
>>> SHARPENED: sigma=4, alpha=3.00
0.649128540305
[[2833 3131]
 [  90 3126]]
>>> SHARPENED: sigma=5, alpha=0.50
0.815904139434
[[4689 1275]
 [ 415 2801]]
>>> SHARPENED: sigma=5, alpha=1.00
0.77037037037
[[4093 1871]
 [ 237 2979]]
>>> SHARPENED: sigma=5, alpha=1.50
0.739978213508
[[3704 2260]
 [ 127 3089]]
>>> SHARPENED: sigma=5, alpha=2.00
0.704575163399
[[3351 2613]
 [  99 3117]]
>>> SHARPENED: sigma=5, alpha=2.50
0.684640522876
[[3154 2810]
 [  85 3131]]
>>> SHARPENED: sigma=5, alpha=3.00
0.651525054466
[[2862 3102]
 [  97 3119]]
>>> SHARPENED: sigma=6, alpha=0.50
0.816339869281
[[4698 1266]
 [ 420 2796]]
>>> SHARPENED: sigma=6, alpha=1.00
0.775272331155
[[4133 1831]
 [ 232 2984]]
>>> SHARPENED: sigma=6, alpha=1.50
0.744117647059
[[3749 2215]
 [ 134 3082]]
>>> SHARPENED: sigma=6, alpha=2.00
0.708278867102
[[3390 2574]
 [ 104 3112]]
>>> SHARPENED: sigma=6, alpha=2.50
0.685076252723
[[3168 2796]
 [  95 3121]]
>>> SHARPENED: sigma=6, alpha=3.00
0.660675381264
[[2946 3018]
 [  97 3119]]
>>> SHARPENED: sigma=7, alpha=0.50
0.816448801743
[[4712 1252]
 [ 433 2783]]
>>> SHARPENED: sigma=7, alpha=1.00
0.773965141612
[[4156 1808]
 [ 267 2949]]
>>> SHARPENED: sigma=7, alpha=1.50
0.748583877996
[[3807 2157]
 [ 151 3065]]
>>> SHARPENED: sigma=7, alpha=2.00
0.710675381264
[[3428 2536]
 [ 120 3096]]
>>> SHARPENED: sigma=7, alpha=2.50
0.688453159041
[[3203 2761]
 [  99 3117]]
>>> SHARPENED: sigma=7, alpha=3.00
0.664488017429
[[2989 2975]
 [ 105 3111]]

Process finished with exit code 0
'''

