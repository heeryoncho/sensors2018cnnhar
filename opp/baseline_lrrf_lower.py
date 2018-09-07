from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import select_data as sd


'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code outputs the LOWER body sensors data HAR performance using 
other baseline machine learning techniques, such as 
logistic regression and random forest, 
given in the bar graph of Figure 15 (red bars indicating Lower Body Sensors). 
(Sensors 2018, 18(4), 1055, page 17 of 24)

'''


print "========================================================="
print "   Outputs performance of other ML techniques, namely,"
print "   Logistic Regression & Random Forest"
print "   Using LOWER body sensors data."
print "========================================================="

print "\n==========================="
print "      [LOWER] 4-Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "end2end")

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
print "  [LOWER] Abstract Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "abst")

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
print "     [LOWER] UP Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "up")

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
print "    [LOWER] DOWN Class"
print "===========================\n"

X_train, y_train, X_valid, y_valid, X_test, y_test = sd.load_data("lower", "down")

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
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/baseline_lrrf_lower.py
=========================================================
   Outputs performance of other ML techniques, namely,
   Logistic Regression & Random Forest
   Using LOWER body sensors data.
=========================================================

===========================
      [LOWER] 4-Class
===========================

--- Logistic Regression ---
Test Acc:  0.894607843137
[[4947  283   96    0]
 [ 953 2860   71    1]
 [  13    2 3445    0]
 [   0    0    0  793]] 


------ Random Forest ------
Test Acc:  0.893122400475
[[5221  105    0    0]
 [1038 2847    0    0]
 [  47    0 3413    0]
 [  40   14  195  544]]

===========================
  [LOWER] Abstract Class
===========================

--- Logistic Regression ---
Test ACC:  0.734625668449
[[9098  113]
 [3460  793]] 

------ Random Forest ------
Test Acc:  0.988042186572
[[9211    0]
 [ 161 4092]]

===========================
     [LOWER] UP Class
===========================

--- Logistic Regression ---
Test Acc:  0.862881337531
[[5070  256]
 [1007 2878]] 

------ Random Forest ------
Test Acc:  0.858430137879
[[5178  148]
 [1156 2729]]

===========================
    [LOWER] DOWN Class
===========================

--- Logistic Regression ---
Test Acc:  1.0
[[3460    0]
 [   0  793]] 

------ Random Forest ------
Test Acc:  0.959322830943
[[3460    0]
 [ 173  620]]

--- End Output ---

Process finished with exit code 0
'''