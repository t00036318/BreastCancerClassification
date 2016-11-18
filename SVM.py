import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)   #Dataframe: conjunto de datos


X, y = df.iloc[:, 2:31], df.iloc[:, 1]  #DataFrame con los valores calculados para cada muestra y Series lasificacion correspondiente de cada fila en X

y.replace('M', -1, inplace=True)
y.replace('B', 1, inplace=True)


linear_svc = SVC(kernel='linear')                           #Kernel lineal
poly_svc = SVC(kernel='poly', gamma=0.2, degree=2)          #Kernel polinomial
rbf_svc = SVC(kernel='rbf', gamma=0.0001, C=6.0)            #Kernel gaussiano
sig_svc = SVC(kernel='sigmoid', gamma=0.0000001, C=8.0)     #Kernel sigmoide

kernels = [linear_svc, poly_svc, rbf_svc, sig_svc]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)         #len(X_train) = 284, len(y_train) = 284
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)    #len(X_test) = 143, len(y_test) = 143, len(X_val) = 142, len(y_val) = 142


def plot_confusion_matrix(confmat):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues,  alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()


def train_and_evaluate(clf, x_train, x_val, x_test, y_train, y_test, y_val):
    clf.fit(x_train, y_train)                           #Entrenamiento del modelo
    y_pred = clf.predict(x_val)                         #Predicciones del modelo entrenado

    error = metrics.mean_absolute_error(y_val, y_pred)
    print(error*100)

    y_pred = clf.predict(x_test)
    test(y_pred, y_test)


def test(preds, y_test):
    print("Reporte de Clasificación:")
    print(metrics.classification_report(y_test, preds))
    print("Matriz de Confusión:")
    print(metrics.confusion_matrix(y_test, preds))
    #plot_confusion_matrix(metrics.confusion_matrix(y_test, preds))

for k in kernels:
    train_and_evaluate(k, X_train, X_val, X_test, y_train, y_test, y_val)


# -------------------------------------------------------------------------------------------------

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train, y_train)