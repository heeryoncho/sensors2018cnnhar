import numpy as np
import process_data


# Load all train and test data labels (* dynamic and static data are mixed.)

y_train = process_data.load_y("train")
y_test = process_data.load_y("test")

print "\nLabel (i.e., activity) constitution:"
print "1: Walking, 2: WU, 3: WD"
print "4: Sitting, 5: Standing, 6: Laying"

unique, counts = np.unique(y_train, return_counts=True)
print "\nTrain data label statistics:"
print np.asarray((unique, counts)).T

unique, counts = np.unique(y_test, return_counts=True)
print "\nTest data label statistics::"
print np.asarray((unique, counts)).T


'''
/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/har/ref_har_data_labels.py

Label (i.e., activity) constitution:
1: Walking, 2: WU, 3: WD
4: Sitting, 5: Standing, 6: Laying

Train data label statistics:
[[   1 1226]
 [   2 1073]
 [   3  986]
 [   4 1286]
 [   5 1374]
 [   6 1407]]

Test data label statistics::
[[  1 496]
 [  2 471]
 [  3 420]
 [  4 491]
 [  5 532]
 [  6 537]]

Process finished with exit code 0

'''
