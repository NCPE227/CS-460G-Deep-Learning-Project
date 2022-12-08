# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:58:00 2022

@author: Dallas Garriott
Machine Learning Citrus Leaves Group Project - Metrics
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from math import sqrt
import numpy
from sklearn import metrics

#confusion matrix = predicted vs actual... Accuracy = (true+ + true-)/totalsample
actual = numpy.random.binomial(1,.9,size = 1000)
predictions = numpy.random.binomial(1,.9,size = 1000)

confusionmatrix = metrics.confusion_matrix(actual, predictions)

display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionmatrix, display_labels = [False, True])
print("Confusion Matrix")
print("   FF FT")
print(confusionmatrix)
print("   TF TT")

#accuracy=(number of correct predictions)/(total number of predictions made)
print("Accuracy: %.3f" % accuracy_score(actual, predictions))

#Mean squared error = (1/N) * summation from j=1 to N (ysubj - ynaughtsubj)^2
def mserror(actual,predictions):
    sum_error = 0.000
    for i in range(len(actual)):
        predict_error = predictions[i]-actual[i]
        sum_error += (predict_error ** 2)
    smean_error = sum_error / float(len(actual))
    return smean_error
print("Mean squared error: %.3f" % mserror(actual, predictions))
print("Root mean squared error: %.3f" % sqrt(mserror(actual,predictions)))


#Mean absolute error = (1/N) * summation from j=1 to N (absolutevalue(ysubj-ynaughtj))
def maerror(actual,predictions):
    sum_error = 0.000
    for i in range(len(actual)):
        sum_error += abs(actual[i]-predictions[i])
    amean_error = sum_error / float(len(actual))
    return amean_error
print("Mean absolute error: %.3f" % maerror(actual,predictions))
    

#F1 Score = 2* (1/((1/precision)+(1/recall)))
print("F1 Score Micro: %.3f" % f1_score(actual, predictions, average= "micro"))
print("F1 Score Macro: %.3f" % f1_score(actual, predictions, average= "macro"))
print("F1 Score Weighted: %.3f" % f1_score(actual, predictions, average= "weighted"))


#precision= true+/(true+ + false+); recall= true+/(true+ + false-)
print("Precision Micro: %.3f" % precision_score(actual, predictions, average= "micro"))
print("Precision Macro: %.3f" % precision_score(actual, predictions, average= "macro"))
print("Precision Weighted: %.3f" % precision_score(actual, predictions, average= "weighted"))

print("Recall Micro: %.3f" % recall_score(actual, predictions, average= "micro"))
print("Recall Macro: %.3f" % recall_score(actual, predictions, average= "macro"))
print("Recall Weighted: %.3f" % recall_score(actual, predictions, average= "weighted"))
