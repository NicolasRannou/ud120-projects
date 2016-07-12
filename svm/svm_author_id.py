#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]

#########################################################
def SVM():
  
    from sklearn.svm import SVC
    clf = SVC( kernel="rbf", C=10000.0 )

    t0 = time()
    clf.fit( features_train, labels_train )
    print "fitting time:", round(time()-t0, 3), "s"

    t0 = time()
    prediction = clf.predict( features_test )
    print "prediction time:", round(time()-t0, 3), "s"
    print "prediction 10:", prediction[10]
    print "prediction 26:", prediction[26]
    print "prediction 50:", prediction[50]

    chris = 0
    for user in prediction:
        if user == 1:
            chris += 1
    
    print "predictions for chris", chris
       
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score( prediction, labels_test )
    print "accuracy:", accuracy 
#########################################################

SVM()

