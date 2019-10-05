#Author: Marshal Will
#Program takes testing and trainng data and outputs the performance metrics
#The data set being imported
# contains targets and outputs from a machine learning algorithm
# after leave-one-out, 10-fold validation.
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return (TP, FP, TN ,FN)

#Method reads in the CSV
def get_data_csv():
    with open('TestingMetrics.csv') as f:
        reader = csv.reader(f)
        list_data = list(reader)
        list_data_df = pd.DataFrame(data=list_data)
    return list_data_df
#Method sets the predictor variables for variable
def set_pred(df,loc):
    y_pred = df.iloc[loc, 1:21].copy()
    return y_pred
#Method sets a actual/true values for variable
def set_true(df,loc):
    y_actual = df.iloc[loc + 1, 1:21].copy()
    return y_actual

#Method calculates basic metrics based on the Confusion Matrix
def set_plot(cfm, location):
    FP = cfm.sum(axis=0) - np.diag(cfm)
    FN = cfm.sum(axis=1) - np.diag(cfm)
    TP = np.diag(cfm)
    TN = cfm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    #plot_roc_curve(FPR,TPR, location)#Calls plot method to print ROC curve

    return sum(TP), sum(FP), sum(TN), sum(FN)

#Method prints the ROC curve for each plot
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.suptitle('ROC Curve ' , fontsize=12)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    return

#Method prints all metrics
def print_metrics(TP, FP, TN, FN):
    #True Positive
    print("True Positive")
    print(TP)
    #False Positve
    print("False Positive")
    print(FP)
    #True Negative
    print("True Negative")
    print(TN)
    #False Negative
    print("False Negative")
    print(FN)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print("Accuracy")
    print(ACC)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    print("Sensitivity")
    print(TPR)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    print("Specificity")
    print(TNR)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    print("Precision")
    print(PPV)
    # Recall
    print("Recall")
    print(TPR)
    # F1 Score
    F1 = TP / (TP + (FN + FP)/2)
    print("F1 Score")
    print(F1)
    return

#Returns rates for ROC curve
def return_rates(TP, FP, TN, FN):
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return FPR, TPR

#Method Calculates confusion Matrix
def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return m

#Method cycles through all the Target and True Data Points
def loop_through_csv(df):
    FPR = []
    TPR = []
    n = 1 # n stores number of each run
    #for loop runs through point in the data set
    for i in range(1,29,3):
        y_pred = set_pred(df, i).copy()#sets the predictor value
        y_true = set_true(df, i).copy()#sets the true value
        #prints which run it is
        print("Run "+ str(n))
        #Use Sklearn to check if matrix and summary is correct
        print("Confusion Matrix with summary using Sklearn to Check")
        print(cm(y_true, y_pred))
        cfm = cm(y_true, y_pred)
        print(classification_report(y_true, y_pred))
        #Field is calculated without library
        print("Calculated Confusion Matrix with Summary that is Calculated")
        print(pd.crosstab(y_true, y_pred))
        #Prints the the confusion matrix
        print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
        #Returns the
        (TP, FP, TN, FN) = set_plot(cfm, n)
        print_metrics(TP, FP, TN, FN)
        FPRm, TPRm = return_rates(TP, FP, TN, FN)
        FPR.append(FPRm)
        TPR.append(TPRm)
        n = n + 1

    return FPR, TPR

df = get_data_csv()

(FPR,TPR) = loop_through_csv(df)


print(FPR)
print(TPR)
#Plots ROC
plot_roc_curve(FPR,TPR)

