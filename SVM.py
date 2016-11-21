import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)   #Dataframe: conjunto de datos

X, y = df.iloc[:, 2:31], df.iloc[:, 1]                  #DataFrame con los valores calculados para cada muestra y Series correspondiente de cada fila en X

y.replace('M', -1, inplace=True)
y.replace('B', 1, inplace=True)

linear_svc = SVC(kernel='linear')                           #Kernel lineal
poly_svc = SVC(kernel='poly', gamma=0.2, degree=2)          #Kernel polinomial
rbf_svc = SVC(kernel='rbf', gamma=0.0001, C=7.0)            #Kernel gaussiano
sig_svc = SVC(kernel='sigmoid', gamma=0.0000001, C=10.0)     #Kernel sigmoide
kernels = [linear_svc, poly_svc, rbf_svc, sig_svc]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)         #len(X_train) = 284, len(y_train) = 284
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)    #len(X_test) = 143, len(y_test) = 143, len(X_val) = 142, len(y_val) = 142

'''
def plot_confusion_matrix(confmat):                            #Función que grafica la matriz de confusión
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues,  alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()


def train(estimator, x_train, y_train):                         #Función que entrena el modelo
    estimator.fit(x_train, y_train)


#train(rbf_svc, X_train, y_train)
#pred = rbf_svc.predict(X_train)
#print(metrics.classification_report(y_train, pred))


def validate(x_val, y_val):                                  #Función de validación
    mayor, nombre, best_K = 0, "", 0
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

        prom = scores.mean()

        if prom > mayor:
            mayor = prom
            nombre = K
            best_K = k

        print("Coeficiente de determinación promedio del kernel", K,"=", prom)    #Coeficiente: calidad del modelo para replicar los resultados

    print("Mejor kernel:", nombre)
    test(best_K, X_train, y_train, X_test, y_test)


def test(best_K, x_train, y_train, x_test, y_test):
    train(best_K, x_train, y_train)
    preds = best_K.predict(x_test)
    print_results(x_test, y_test, preds)
'''
def print_results(x_test, y_test, preds):
    print("Reporte de Clasificación:")
    print(metrics.classification_report(y_test, preds))
    print("Matriz de Confusión:")
    M = metrics.confusion_matrix(y_test, preds)
    print(M)
    #plot_confusion_matrix(M)
    scores = cross_val_score(best_K, x_test, y_test, cv=5)
    print("Exactitud: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #plot_decision_regions(X_train_plot, y_train, classifier=best_K)
    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc='upper left')
    #plt.show()

#validate(X_val, y_val)

print("-------------------------------------------------------------------------------------------------")


#C = [1, 10, 100, 1000]                         #Valores más pequeños de c incrementan la regularización

lr = LogisticRegression(C=10)
lr.fit(X_train, y_train)
print('Exactitud:', lr.score(X_val, y_val))

f = 5
cv = KFold(len(y_val), f, shuffle=True, random_state=0)
scores = cross_val_score(lr, X_val, y_val, cv=cv)
print("Coeficiente de determinación promedio", scores.mean())


def test(x_train, y_train, x_test, y_test):
    lr = LogisticRegression(C=1, random_state=0)
    lr.fit(x_train, y_train)
    preds = lr.predict(x_test)

    print_results(x_test, y_test, preds)


test(X_train, y_train, X_test, y_test)









'''
def validate_lr(x_train, y_train, x_val, y_val):
    mayor, z, prom = 0, 0, 0
    for c in C:
        lr = LogisticRegression(C=c, random_state=0)
        f = 5
        cv = KFold(len(y_val), f, shuffle=True, random_state=0)
        scores = cross_val_score(lr, x_val, y_val, cv=cv)
        prom = scores.mean()

        if prom > mayor:
            mayor = prom
            z = c

        print("Coeficiente de determinación promedio ", prom)

    test(z, x_train, y_train, X_test, y_test)


def test(best_c, x_train, y_train, x_test, y_test):
    print(best_c)
    lr = LogisticRegression(C=10, random_state=0)
    lr.fit(x_train, y_train)
    preds = lr.predict(x_test)

    print("Reporte de Clasificación:")
    print(metrics.classification_report(y_test, preds))
    print("Matriz de Confusión:")
    M = metrics.confusion_matrix(y_test, preds)
    print(M)
    #plot_confusion_matrix(M)
    scores = cross_val_score(lr, x_test, y_test, cv=5)
    print("Exactitud: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

validate_lr(X_train, y_train, X_val, y_val)
'''