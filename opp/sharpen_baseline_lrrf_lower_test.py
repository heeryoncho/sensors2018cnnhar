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

The performance is measured using LOWER body X_test, y_test dataset.

See right line graphs in Figure 14 (b) & (c) (Test Data Recognition Accuracy). 
(Sensors 2018, 18(4), 1055, page 16 of 24)
'''

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "up")

print "\n=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ==="
print "===     [LOWER body sensors data]  UP Class     ==="
print "===     Logistic Regression & Random Forest     ==="
print "===           Evaluation on TEST DATA           ===\n"

print "\n============================================"
print "          LOGISTIC REGRESSION"
print "============================================\n"

clf_lr = LogisticRegression(random_state=2018)
clf_lr.fit(X_train, y_train)

print ">>> RAW:"
pred_lr = clf_lr.predict(X_test)
print accuracy_score(y_test, pred_lr)
print confusion_matrix(y_test, pred_lr), '\n'

alpha = np.arange(0.5, 3.5, 0.5)
sigma = np.arange(3, 8, 1)

for s in sigma:
    for a in alpha:
        x_test_sharpen = sd.sharpen(X_test, s, a)
        pred_lr = clf_lr.predict(x_test_sharpen)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_test, pred_lr)
        print confusion_matrix(y_test, pred_lr)


print "\n============================================"
print "               RANDOM FOREST"
print "============================================\n"

clf_rf = RandomForestClassifier(random_state=2018, max_depth=5, n_estimators=10, max_features=1)
clf_rf.fit(X_train, y_train)

print ">>> RAW:"
pred_rf = clf_rf.predict(X_test)
print accuracy_score(y_test, pred_rf)
print confusion_matrix(y_test, pred_rf), '\n'

for s in sigma:
    for a in alpha:
        x_test_sharpen = sd.sharpen(X_test, s, a)
        pred_rf = clf_rf.predict(x_test_sharpen)
        print ">>> SHARPENED: sigma={}, alpha={:.2f}".format(s, a)
        print accuracy_score(y_test, pred_rf)
        print confusion_matrix(y_test, pred_rf)



'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/sharpen_baseline_lrrf_lower_test.py

=== COMPARE ACCURACY: NO SHARPEN vs. SHARPENED  ===
===     [LOWER body sensors data]  UP Class     ===
===     Logistic Regression & Random Forest     ===
===           Evaluation on TEST DATA           ===


============================================
          LOGISTIC REGRESSION
============================================

>>> RAW:
0.862881337531
[[5070  256]
 [1007 2878]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.866246878732
[[4658  668]
 [ 564 3321]]
>>> SHARPENED: sigma=3, alpha=1.00
0.857887308653
[[4599  727]
 [ 582 3303]]
>>> SHARPENED: sigma=3, alpha=1.50
0.855715991749
[[4583  743]
 [ 586 3299]]
>>> SHARPENED: sigma=3, alpha=2.00
0.855933123439
[[4583  743]
 [ 584 3301]]
>>> SHARPENED: sigma=3, alpha=2.50
0.855824557594
[[4578  748]
 [ 580 3305]]
>>> SHARPENED: sigma=3, alpha=3.00
0.855173162523
[[4574  752]
 [ 582 3303]]
>>> SHARPENED: sigma=4, alpha=0.50
0.86787536641
[[4691  635]
 [ 582 3303]]
>>> SHARPENED: sigma=4, alpha=1.00
0.859407230485
[[4641  685]
 [ 610 3275]]
>>> SHARPENED: sigma=4, alpha=1.50
0.856475952665
[[4622  704]
 [ 618 3267]]
>>> SHARPENED: sigma=4, alpha=2.00
0.854304635762
[[4611  715]
 [ 627 3258]]
>>> SHARPENED: sigma=4, alpha=2.50
0.854196069916
[[4608  718]
 [ 625 3260]]
>>> SHARPENED: sigma=4, alpha=3.00
0.853544674845
[[4604  722]
 [ 627 3258]]
>>> SHARPENED: sigma=5, alpha=0.50
0.867006839648
[[4690  636]
 [ 589 3296]]
>>> SHARPENED: sigma=5, alpha=1.00
0.861144284008
[[4649  677]
 [ 602 3283]]
>>> SHARPENED: sigma=5, alpha=1.50
0.858647269569
[[4646  680]
 [ 622 3263]]
>>> SHARPENED: sigma=5, alpha=2.00
0.857778742808
[[4641  685]
 [ 625 3260]]
>>> SHARPENED: sigma=5, alpha=2.50
0.856693084356
[[4640  686]
 [ 634 3251]]
>>> SHARPENED: sigma=5, alpha=3.00
0.856041689285
[[4639  687]
 [ 639 3246]]
>>> SHARPENED: sigma=6, alpha=0.50
0.866355444577
[[4690  636]
 [ 595 3290]]
>>> SHARPENED: sigma=6, alpha=1.00
0.860710020628
[[4658  668]
 [ 615 3270]]
>>> SHARPENED: sigma=6, alpha=1.50
0.857018781891
[[4643  683]
 [ 634 3251]]
>>> SHARPENED: sigma=6, alpha=2.00
0.855715991749
[[4640  686]
 [ 643 3242]]
>>> SHARPENED: sigma=6, alpha=2.50
0.855173162523
[[4639  687]
 [ 647 3238]]
>>> SHARPENED: sigma=6, alpha=3.00
0.854521767452
[[4637  689]
 [ 651 3234]]
>>> SHARPENED: sigma=7, alpha=0.50
0.867115405493
[[4698  628]
 [ 596 3289]]
>>> SHARPENED: sigma=7, alpha=1.00
0.859732928021
[[4654  672]
 [ 620 3265]]
>>> SHARPENED: sigma=7, alpha=1.50
0.855390294213
[[4635  691]
 [ 641 3244]]
>>> SHARPENED: sigma=7, alpha=2.00
0.854087504071
[[4632  694]
 [ 650 3235]]
>>> SHARPENED: sigma=7, alpha=2.50
0.853436109
[[4631  695]
 [ 655 3230]]
>>> SHARPENED: sigma=7, alpha=3.00
0.853001845619
[[4630  696]
 [ 658 3227]]

============================================
               RANDOM FOREST
============================================

>>> RAW:
0.858430137879
[[5178  148]
 [1156 2729]] 

>>> SHARPENED: sigma=3, alpha=0.50
0.886548691782
[[4940  386]
 [ 659 3226]]
>>> SHARPENED: sigma=3, alpha=1.00
0.876777765715
[[4616  710]
 [ 425 3460]]
>>> SHARPENED: sigma=3, alpha=1.50
0.856041689285
[[4311 1015]
 [ 311 3574]]
>>> SHARPENED: sigma=3, alpha=2.00
0.838671154055
[[4132 1194]
 [ 292 3593]]
>>> SHARPENED: sigma=3, alpha=2.50
0.819454999457
[[3937 1389]
 [ 274 3611]]
>>> SHARPENED: sigma=3, alpha=3.00
0.799153186408
[[3716 1610]
 [ 240 3645]]
>>> SHARPENED: sigma=4, alpha=0.50
0.883400282271
[[4840  486]
 [ 588 3297]]
>>> SHARPENED: sigma=4, alpha=1.00
0.870263815004
[[4529  797]
 [ 398 3487]]
>>> SHARPENED: sigma=4, alpha=1.50
0.852350450548
[[4253 1073]
 [ 287 3598]]
>>> SHARPENED: sigma=4, alpha=2.00
0.833785691022
[[4035 1291]
 [ 240 3645]]
>>> SHARPENED: sigma=4, alpha=2.50
0.818803604386
[[3851 1475]
 [ 194 3691]]
>>> SHARPENED: sigma=4, alpha=3.00
0.799478883943
[[3664 1662]
 [ 185 3700]]
>>> SHARPENED: sigma=5, alpha=0.50
0.879166214309
[[4807  519]
 [ 594 3291]]
>>> SHARPENED: sigma=5, alpha=1.00
0.866572576268
[[4499  827]
 [ 402 3483]]
>>> SHARPENED: sigma=5, alpha=1.50
0.853327543155
[[4228 1098]
 [ 253 3632]]
>>> SHARPENED: sigma=5, alpha=2.00
0.832591466725
[[4016 1310]
 [ 232 3653]]
>>> SHARPENED: sigma=5, alpha=2.50
0.814026707198
[[3811 1515]
 [ 198 3687]]
>>> SHARPENED: sigma=5, alpha=3.00
0.800238844859
[[3661 1665]
 [ 175 3710]]
>>> SHARPENED: sigma=6, alpha=0.50
0.879709043535
[[4822  504]
 [ 604 3281]]
>>> SHARPENED: sigma=6, alpha=1.00
0.867983932255
[[4523  803]
 [ 413 3472]]
>>> SHARPENED: sigma=6, alpha=1.50
0.853110411465
[[4255 1071]
 [ 282 3603]]
>>> SHARPENED: sigma=6, alpha=2.00
0.834111388557
[[4031 1295]
 [ 233 3652]]
>>> SHARPENED: sigma=6, alpha=2.50
0.81826077516
[[3861 1465]
 [ 209 3676]]
>>> SHARPENED: sigma=6, alpha=3.00
0.802844425144
[[3703 1623]
 [ 193 3692]]
>>> SHARPENED: sigma=7, alpha=0.50
0.879274780154
[[4827  499]
 [ 613 3272]]
>>> SHARPENED: sigma=7, alpha=1.00
0.869286722397
[[4550  776]
 [ 428 3457]]
>>> SHARPENED: sigma=7, alpha=1.50
0.857778742808
[[4316 1010]
 [ 300 3585]]
>>> SHARPENED: sigma=7, alpha=2.00
0.834219954402
[[4048 1278]
 [ 249 3636]]
>>> SHARPENED: sigma=7, alpha=2.50
0.816849419173
[[3875 1451]
 [ 236 3649]]
>>> SHARPENED: sigma=7, alpha=3.00
0.802844425144
[[3726 1600]
 [ 216 3669]]

Process finished with exit code 0
'''

