from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import select_data as sd


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code outputs the raw sensor data HAR performance using 
other baseline machine learning techniques, such as 
logistic regression and random forest, 
given in the bar graph of Figure 16 (blue bars indicating Raw Time Series data). 
(Sensors 2018, 18(4), 1055, page 17 of 24)

'''


print "========================================================="
print "   Outputs performance of other ML techniques, namely,"
print "   Logistic Regression & Random Forest"
print "   Using RAW sensor data."
print "========================================================="

print "\n==========================="
print "      [UPPER] 4-Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("raw", "end2end")

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

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("raw", "abst")

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

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("raw", "up")

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

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("raw", "down")

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
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/baseline_lrrf_raw.py
=========================================================
   Outputs performance of other ML techniques, namely,
   Logistic Regression & Random Forest
   Using RAW sensor data.
=========================================================

===========================
      [UPPER] 4-Class
===========================

--- Logistic Regression ---
Test Acc:  0.856431966726
[[5094  218   14    0]
 [1598 2219   54   14]
 [  35    0 3425    0]
 [   0    0    0  793]] 


------ Random Forest ------
Test Acc:  0.721628045157
[[5121  202    3    0]
 [1592 2292    1    0]
 [1941    2 1517    0]
 [   2    2    3  786]]

===========================
  [UPPER] Abstract Class
===========================

--- Logistic Regression ---
Test ACC:  0.785427807487
[[9153   58]
 [2831 1422]] 

------ Random Forest ------
Test Acc:  0.743092691622
[[9211    0]
 [3459  794]]

===========================
     [UPPER] UP Class
===========================

--- Logistic Regression ---
Test Acc:  0.815763760721
[[5092  234]
 [1463 2422]] 

------ Random Forest ------
Test Acc:  0.806969927261
[[5120  206]
 [1572 2313]]

===========================
    [UPPER] DOWN Class
===========================

--- Logistic Regression ---
Test Acc:  1.0
[[3460    0]
 [   0  793]] 

------ Random Forest ------
Test Acc:  0.998824359276
[[3460    0]
 [   5  788]]

--- End Output ---

Process finished with exit code 0
'''