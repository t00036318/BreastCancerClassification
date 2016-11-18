import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)   #Dataframe: conjunto de datos


X, y = df.iloc[:, 2:31], df.iloc[:, 1]  #DataFrame con los valores calculados para cada muestra y Series lasificacion correspondiente de cada fila en X

#y.replace('M', -1, inplace=True)
#y.replace('B', 1, inplace=True)


linear_svc = SVC(kernel='linear')     #Kernel lineal
pol_svc = SVC(kernel='poly')          #Kernel polinomial
rbf_svc = SVC(kernel='rbf')           #Kernel gaussiano
sig_svc = SVC(kernel='sigmoid')       #Kernel sigmoide

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)         #len(X_train) = 284, len(y_train) = 284
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)    #len(X_test) = 143, len(y_test) = 143, len(X_val) = 142, len(y_val) = 142

'''
def train(svc, X, y):
    svc.fit(X, y)

train(linear_svc,X_train, y_train)
'''


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Reporte de clasificación:")
    print(metrics.classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    maxConf = metrics.confusion_matrix(y_test, y_pred)
    print(maxConf)
    #sensitivity(maxConf)

'''
def sensitivity(m):
    tn = m[0, 1]    #True Negatives
    fp = m[1, 0]    #False Positives
    spc = tn / (tn + fp)
    print(spc)
'''

train_and_evaluate(linear_svc, X_train, X_test, y_train, y_test)
#train(pol_svc,X_train, y_train)
#train(rbf_svc,X_train, y_train)
#train(sig_svc,X_train, y_train)
