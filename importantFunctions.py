import numpy as np # linear algebra
import pandas as pd 


#Visualizations library
import seaborn as sns 
import matplotlib.pyplot as plt
import graphviz 

# Model selection and evaluation
from sklearn.model_selection import KFold, cross_val_score, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support

import itertools

# It can be used to print dataset information.
def printDataInfo(data):
    print("Dataset Shape:")
    print(data.shape)
    print("\n")
    print("Dataset Columns\t\t    Features:")
    print(data.dtypes)

# Dataframe that shows Null count and percent of null counts of all features.
def checkNull(dataFrame):
    nullCount = dataFrame.isnull().sum().sort_values(ascending=False)
    nullCount = nullCount[nullCount != 0]
    nullPercent = ((nullCount)/dataFrame.shape[0]) * 100
    nullCols = pd.DataFrame([nullCount, nullPercent], index=['Count', 'Percent']).transpose()
    return nullCols

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def pred2DF(pred, y_test):
    prediction = pd.Series(pred)
    prediction.index = y_test.index
    df = pd.DataFrame({'Op': y_test, 'Prediction': prediction}, index=y_test.index)
    # df = df.join(sampleCodes, how='inner')
    return df

def calcAccuracy(op, pred):
    return (op == pred).sum()/pred.shape[0]

def calcError(op, pred):
    return (op != pred).sum()/pred.shape[0]

def makePredict(model, mName, x_train, y_train, x_test, y_test):
    trDf = crossValidate(model, x_train,  y_train, n=5)
    model.fit(x_train, y_train)
    trPredict = trDf['Pred'].astype(int).ravel()
    y_train = trDf['Op'].astype(int).ravel()
    tePredict = model.predict(x_test)
    tePredict = tePredict.ravel()
    df = pred2DF(tePredict, y_test)
    
    prec, rec, f1score, sup = precision_recall_fscore_support(y_test, tePredict, average='binary')
    acc = calcAccuracy(y_test, tePredict)
    err = calcError(y_test, tePredict)
    prec = round(prec, 2)
    rec = round(rec, 2)
    f1score = round(f1score, 2)
    acc = round(acc, 2)
    err = round(err, 2)
    
    precTr, recTr, f1scoreTr, supTr = precision_recall_fscore_support(y_train, trPredict, average='binary')
    accTr = calcAccuracy(y_train, trPredict)
    errTr = calcError(y_train, trPredict)
    precTr = round(precTr, 2)
    recTr = round(recTr, 2)
    f1scoreTr = round(f1scoreTr, 2)
    accTr = round(accTr, 2)
    errTr = round(errTr, 2)
    
    print(mName, (9 - len(mName)) * ' ', '\t\tTest\t\tTrain(K-Fold)')
    print()
    print("Accuracy \t\t", acc, '\t\t', accTr)
    print("Error    \t\t", err, '\t\t', errTr)
    print("Recall   \t\t", rec, '\t\t', recTr)
    print("Precision\t\t", prec,'\t\t', precTr)
    print("F1 Score \t\t", f1score, '\t\t', f1scoreTr)
    print()
    print("Classification Report of ", mName, " on test : ")
    cm = confusion_matrix(y_test, tePredict, labels=[0, 1])
    cm1 = confusion_matrix(y_train, trPredict, labels=[0, 1])
    plt.figure(figsize=(16,8))
    
    plt.subplot(1,2, 1)
    plotConfusionMatrix(cm, classes=['Benign(0)', 'Maligant(1)'],title='Test Confusion matrix')
    
    plt.subplot(1,2, 2)
    plotConfusionMatrix(cm1, classes=['Benign(0)', 'Maligant(1)'],title='Train Confusion matrix')
    
    plt.tight_layout()
    plt.show()
    return df, tePredict

def crossValidate(clf, X, y, n=3):
    kf = KFold(n_splits=n)
    df = pd.DataFrame(columns=['Pred', 'Op'])
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    for trInd, teInd in kf.split(X):
        X_tr, X_te = X.iloc[trInd,:], X.iloc[teInd,:]
        y_tr, y_te = y.iloc[trInd], y.iloc[teInd]
        clf.fit(X_tr, y_tr)
        pred = pd.Series(clf.predict(X_te), index=teInd, name='Pred')
        y_te = pd.Series(y_te, index=teInd, name='Op')
        tempDf = pd.concat([pred, y_te], axis=1)
        df = df.append(tempDf)
    return df 