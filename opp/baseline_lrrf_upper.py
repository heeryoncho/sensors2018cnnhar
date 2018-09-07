from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import select_data as sd


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code outputs the UPPER body sensors data HAR performance using 
other baseline machine learning techniques, such as 
logistic regression and random forest, 
given in the bar graph of Figure 15 (blue bars indicating Upper Body Sensors). 
(Sensors 2018, 18(4), 1055, page 17 of 24)

'''


print "========================================================="
print "   Outputs performance of other ML techniques, namely,"
print "   Logistic Regression & Random Forest"
print "   Using UPPER body sensors data."
print "========================================================="

print "\n==========================="
print "      [UPPER] 4-Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("upper", "end2end")

clf_lr = LogisticRegression(random_state=2018)
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
print "--- Logistic Regression ---"
print "Test Acc: ", accuracy_score(y_test, pred_lr)
print confusion_matrix(y_test, pred_lr), '\n'

clf_dt = RandomForestClassifier(random_state=2018, max_depth=5, n_estimators=10, max_features=1)
clf_dt.fit(X_train, y_train)
pred_dt = clf_dt.predict(X_test)
print "\n------ Random Forest ------"
print "Test Acc: ", accuracy_score(y_test, pred_dt)
print confusion_matrix(y_test, pred_dt)

#---------------------------------------------

print "\n==========================="
print "  [UPPER] Abstract Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("upper", "abst")

clf_lr = LogisticRegression(random_state=2018)
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
print "--- Logistic Regression ---"
print "Test ACC: ", accuracy_score(y_test, pred_lr)
print confusion_matrix(y_test, pred_lr), '\n'

clf_dt = RandomForestClassifier(random_state=2018, max_depth=5, n_estimators=10, max_features=1)
clf_dt.fit(X_train, y_train)
pred_dt = clf_dt.predict(X_test)
print "------ Random Forest ------"
print "Test Acc: ", accuracy_score(y_test, pred_dt)
print confusion_matrix(y_test, pred_dt)

#---------------------------------------------

print "\n==========================="
print "     [UPPER] UP Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("upper", "up")

clf_lr = LogisticRegression(random_state=2018)
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
print "--- Logistic Regression ---"
print "Test Acc: ", accuracy_score(y_test, pred_lr)
print confusion_matrix(y_test, pred_lr), '\n'

clf_dt = RandomForestClassifier(random_state=2018, max_depth=5, n_estimators=10, max_features=1)
clf_dt.fit(X_train, y_train)
pred_dt = clf_dt.predict(X_test)
print "------ Random Forest ------"
print "Test Acc: ", accuracy_score(y_test, pred_dt)
print confusion_matrix(y_test, pred_dt)

#---------------------------------------------

print "\n==========================="
print "    [UPPER] DOWN Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("upper", "down")

clf_lr = LogisticRegression(random_state=2018)
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
print "--- Logistic Regression ---"
print "Test Acc: ", accuracy_score(y_test, pred_lr)
print confusion_matrix(y_test, pred_lr), '\n'

clf_dt = RandomForestClassifier(random_state=2018, max_depth=5, n_estimators=10, max_features=1)
clf_dt.fit(X_train, y_train)
pred_dt = clf_dt.predict(X_test)
print "------ Random Forest ------"
print "Test Acc: ", accuracy_score(y_test, pred_dt)
print confusion_matrix(y_test, pred_dt)


print "\n--- End Output ---"


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/baseline_lrrf_upper.py
=========================================================
   Outputs performance of other ML techniques, namely,
   Logistic Regression & Random Forest
   Using UPPER body sensors data.
=========================================================

===========================
      [UPPER] 4-Class
===========================

--- Logistic Regression ---
Test Acc:  0.833184789067
[[4860  333  133    0]
 [1379 2497    9    0]
 [ 316   76 3068    0]
 [   0    0    0  793]] 


------ Random Forest ------
Test Acc:  0.80830362448
[[4959  218  149    0]
 [1620 2199   66    0]
 [  32   12 3416    0]
 [   9    0  475  309]]

===========================
  [UPPER] Abstract Class
===========================

--- Logistic Regression ---
Test ACC:  0.973336304219
[[9131   80]
 [ 279 3974]] 

------ Random Forest ------
Test Acc:  0.982174688057
[[9176   35]
 [ 205 4048]]

===========================
     [UPPER] UP Class
===========================

--- Logistic Regression ---
Test Acc:  0.812289653675
[[4875  451]
 [1278 2607]] 

------ Random Forest ------
Test Acc:  0.809358375855
[[5064  262]
 [1494 2391]]

===========================
    [UPPER] DOWN Class
===========================

--- Logistic Regression ---
Test Acc:  1.0
[[3460    0]
 [   0  793]] 

------ Random Forest ------
Test Acc:  0.981189748413
[[3460    0]
 [  80  713]]

--- End Output ---

Process finished with exit code 0

'''