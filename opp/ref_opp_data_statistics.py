import numpy as np

'''
See paper:  Sensors 2018, 18(4), 1055; https://doi.org/10.3390/s18041055
"Divide and Conquer-Based 1D CNN Human Activity Recognition Using Test Data Sharpening"
by Heeryon Cho & Sang Min Yoon

This code checks the data size & activity label constitution of the 
Opportunity UCI Dataset data used in the experiment.

NOTE:
[Label]                  [Activity]   
 1   -   Locomotion   -   Stand
 2   -   Locomotion   -   Walk
 4   -   Locomotion   -   Sit
 5   -   Locomotion   -   Lie
'''

dir_path = '../data/OpportunityUCIDataset/processed/'


print "-------------------------------------------------"
print "   LOWER BODY SENSORS DATA"
print "-------------------------------------------------"

X_train = np.load(dir_path + "lower_train_X.npy")
y_train = np.load(dir_path + "lower_train_y.npy")

X_valid = np.load(dir_path + "lower_valid_X.npy")
y_valid = np.load(dir_path + "lower_valid_y.npy")

X_test = np.load(dir_path + "lower_test_X.npy")
y_test = np.load(dir_path + "lower_test_y.npy")

print "lower_train_X: ", X_train.shape
print "lower_valid_X: ", X_valid.shape
print "lower_test_X: ", X_test.shape

unique, counts = np.unique(y_train, return_counts=True)
print "--- lower_train_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'

unique, counts = np.unique(y_valid, return_counts=True)
print "--- lower_valid_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'

unique, counts = np.unique(y_test, return_counts=True)
print "--- lower_test_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'


print "-------------------------------------------------"
print "   UPPER BODY SENSORS DATA"
print "-------------------------------------------------"

X_train = np.load(dir_path + "upper_train_X.npy")
y_train = np.load(dir_path + "upper_train_y.npy")

X_valid = np.load(dir_path + "upper_valid_X.npy")
y_valid = np.load(dir_path + "upper_valid_y.npy")

X_test = np.load(dir_path + "upper_test_X.npy")
y_test = np.load(dir_path + "upper_test_y.npy")

print "upper_train_X: ", X_train.shape
print "upper_valid_X: ", X_valid.shape
print "upper_test_X: ", X_test.shape

unique, counts = np.unique(y_train, return_counts=True)
print "--- upper_train_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'

unique, counts = np.unique(y_valid, return_counts=True)
print "--- upper_valid_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'

unique, counts = np.unique(y_test, return_counts=True)
print "--- upper_test_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'


print "-------------------------------------------------"
print "   RAW BODY SENSORS DATA"
print "-------------------------------------------------"

X_train = np.load(dir_path + "raw_train_X.npy")
y_train = np.load(dir_path + "raw_train_y.npy")

X_valid = np.load(dir_path + "raw_valid_X.npy")
y_valid = np.load(dir_path + "raw_valid_y.npy")

X_test = np.load(dir_path + "raw_test_X.npy")
y_test = np.load(dir_path + "raw_test_y.npy")

print "raw_train_X: ", X_train.shape
print "raw_valid_X: ", X_valid.shape
print "raw_test_X: ", X_test.shape

unique, counts = np.unique(y_train, return_counts=True)
print "--- raw_train_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'

unique, counts = np.unique(y_valid, return_counts=True)
print "--- raw_valid_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'

unique, counts = np.unique(y_test, return_counts=True)
print "--- raw_test_y: label freq ---"
print np.asarray((unique, counts)).T, '\n'


'''

/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/ref_opp_data_statistics.py
-------------------------------------------------
   LOWER BODY SENSORS DATA
-------------------------------------------------
lower_train_X:  (28938, 156)
lower_valid_X:  (13609, 156)
lower_test_X:  (13464, 156)
--- lower_train_y: label freq ---
[[    1 13250]
 [    2  7403]
 [    4  6874]
 [    5  1411]] 

--- lower_valid_y: label freq ---
[[   1 5964]
 [   2 3216]
 [   4 3766]
 [   5  663]] 

--- lower_test_y: label freq ---
[[   1 5326]
 [   2 3885]
 [   4 3460]
 [   5  793]] 

-------------------------------------------------
   UPPER BODY SENSORS DATA
-------------------------------------------------
upper_train_X:  (28938, 216)
upper_valid_X:  (13609, 216)
upper_test_X:  (13464, 216)
--- upper_train_y: label freq ---
[[    1 13250]
 [    2  7403]
 [    4  6874]
 [    5  1411]] 

--- upper_valid_y: label freq ---
[[   1 5964]
 [   2 3216]
 [   4 3766]
 [   5  663]] 

--- upper_test_y: label freq ---
[[   1 5326]
 [   2 3885]
 [   4 3460]
 [   5  793]] 

-------------------------------------------------
   RAW BODY SENSORS DATA
-------------------------------------------------
raw_train_X:  (28938, 585)
raw_valid_X:  (13609, 585)
raw_test_X:  (13464, 585)
--- raw_train_y: label freq ---
[[    1 13250]
 [    2  7403]
 [    4  6874]
 [    5  1411]] 

--- raw_valid_y: label freq ---
[[   1 5964]
 [   2 3216]
 [   4 3766]
 [   5  663]] 

--- raw_test_y: label freq ---
[[   1 5326]
 [   2 3885]
 [   4 3460]
 [   5  793]] 


Process finished with exit code 0

'''