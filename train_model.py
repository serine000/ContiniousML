import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import utils 
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")
outliers = np.genfromtxt("data/outliers.csv")
target = np.genfromtxt("data/target.csv")

# Fit a model
nu = outliers.shape[0] / target.shape[0]
model = make_pipeline(StandardScaler(), svm.OneClassSVM(kernel='rbf', nu = nu, gamma=0.5))
model.fit(X_train)


preds = model.predict(X_train)
targs = y_train 
test_preds = model.predict(X_test)
test_targs = y_test 
test_acc = metrics.accuracy_score(test_targs, test_preds)
with open("metrics.txt", "w") as outfile:
    # outfile.write("Training Metrics: \n")
    # outfile.write("Training accuracy: " + str(metrics.accuracy_score(targs, preds)) + "\n")
    # outfile.write("Training precision: " + str(metrics.precision_score(targs, preds)) + "\n")
    # outfile.write("Training recall: " + str(metrics.recall_score(targs, preds)) + "\n")
    # outfile.write("Training f1 score: " + str(metrics.f1_score(targs, preds)) + "\n")
    # outfile.write("Training area under curve (AUC): " + str(metrics.roc_auc_score(targs, preds)) + "\n")
    # outfile.write("\n")
    outfile.write("Testing Metrics: \n")
    outfile.write("Testing accuracy: " + str(test_acc) + "\n")
    # outfile.write("Testing precision: " + str(metrics.precision_score(test_targs, test_preds)) + "\n")
    # outfile.write("Testing recall: " + str(metrics.recall_score(test_targs, test_preds)) + "\n")
    # outfile.write("Testing f1 score: " + str(metrics.f1_score(test_targs, test_preds)) + "\n")
    # outfile.write("Testing area under curve (AUC): " + str(metrics.roc_auc_score(test_targs, test_preds)) + "\n")
    # outfile.write("\n")
