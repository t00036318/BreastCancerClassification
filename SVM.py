import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
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

'''
def plot_confusion_matrix(confmat):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues,  alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()
'''


def train(kernel, x_train, y_train):                         #Función que entrena el modelo
    kernel.fit(x_train, y_train)


def validate(x_val, y_val):               #Función de validación
    mayor, nombre = 0, ""
    for k in kernels:
        f = 5
        cv = KFold(len(y_val), f, shuffle=True, random_state=0)
        scores = cross_val_score(k, x_val, y_val, cv=cv)

        if k == linear_svc:
            K = "Lineal"
        elif k == poly_svc:
            K = "Polinomial"
        elif k == rbf_svc:
            K = "Gaussiano"
        else:
            K = "Sigmoide"

        prom = np.mean(scores)

        if prom > mayor:
            mayor = prom
            nombre = K

        print("Coeficiente de determinación promedio del kernel", K,"=", prom)    #Coeficiente: calidad del modelo para replicar los resultados

    print("Mejor kernel:", nombre)


'''
def train_and_evaluate(kernel, x_train, x_val, x_test, y_train, y_test, y_val):
    clf.fit(x_train, y_train)                           #Entrenamiento del modelo
    y_pred = clf.predict(x_val)

    error = metrics.mean_absolute_error(y_val, y_pred)
    print(error*100)

    y_pred = clf.predict(x_test)
    test(y_pred, y_test)
'''


def test(best_K, x_train, y_train, x_test, y_test):
    train(best_K, x_train, y_train)
    preds = best_K.predict(x_test)

    print("Reporte de Clasificación:")
    print(metrics.classification_report(y_test, preds))
    print("Matriz de Confusión:")
    M = metrics.confusion_matrix(y_test, preds)
    print(M)
    #plot_confusion_matrix(M)
    scores = cross_val_score(best_K, x_test, y_test, cv=5)
    print("Exactitud: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

validate(X_val, y_val)

''

# -------------------------------------------------------------------------------------------------

lr = LogisticRegression(C=1000, random_state=0)

print('Logistic Regression score: %f' % lr.fit(X_train, y_train).score(X_val, y_val))
