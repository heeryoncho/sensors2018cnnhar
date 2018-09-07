import process_data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Load all train and test data (* dynamic and static data are mixed.)

X_train = process_data.load_x("train")
y_train = process_data.load_y("train")

X_test = process_data.load_x("test")
y_test = process_data.load_y("test")


print "=================================="
print " ACCURACY OF OTHER ML CLASSIFIERS"
print "=================================="


# Build a logistic regression classifier and predict

clf_lr = LogisticRegression(random_state=0)
clf_lr.fit(X_train, y_train)

pred_lr = clf_lr.predict(X_test)

print "\n--- Logistic Regression Classifier ---"
print accuracy_score(y_test, pred_lr)
print confusion_matrix(y_test, pred_lr)


# Build an SVM classifier and predict

clf_svm = SVC(random_state=0)
clf_svm.fit(X_train, y_train)

pred_svm = clf_svm.predict(X_test)

print "\n--- SVM Classifier ---"
print accuracy_score(y_test, pred_svm)
print confusion_matrix(y_test, pred_svm)


# Build a neural network classifier and predict

clf_nn = MLPClassifier(random_state=0)
clf_nn.fit(X_train, y_train)

pred_nn = clf_nn.predict(X_test)

print "\n--- Neural Network Classifier ---"
print accuracy_score(y_test, pred_nn)
print confusion_matrix(y_test, pred_nn)


# Build a decision tree classifier and predict

clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)

print "\n--- Decision Tree Classifier ---"
print accuracy_score(y_test, pred_dt)
print confusion_matrix(y_test, pred_dt)


'''

/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/har/ref_har_other_ML.py
==================================
 ACCURACY OF OTHER ML CLASSIFIERS
==================================

--- Logistic Regression Classifier ---
0.961995249406
[[494   0   2   0   0   0]
 [ 23 448   0   0   0   0]
 [  4   9 407   0   0   0]
 [  0   4   0 432  55   0]
 [  2   0   0  13 517   0]
 [  0   0   0   0   0 537]]

--- SVM Classifier ---
0.940278249067
[[492   0   4   0   0   0]
 [ 17 452   2   0   0   0]
 [ 13  29 378   0   0   0]
 [  0   2   0 424  65   0]
 [  0   0   0  44 488   0]
 [  0   0   0   0   0 537]]

--- Neural Network Classifier ---
0.937563624024
[[490   1   5   0   0   0]
 [ 37 434   0   0   0   0]
 [  7  24 389   0   0   0]
 [  0   3   0 414  74   0]
 [  0   0   0  18 514   0]
 [  0   0   0   0  15 522]]

--- Decision Tree Classifier ---
0.859518154055
[[448  24  24   0   0   0]
 [ 74 367  30   0   0   0]
 [ 23  46 351   0   0   0]
 [  0   0   0 373 118   0]
 [  0   0   0  75 457   0]
 [  0   0   0   0   0 537]]

Process finished with exit code 0


'''