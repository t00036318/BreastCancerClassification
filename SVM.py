import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('wdbc.data.txt', header=None)   #Dataframe: conjunto de datos

X, y = df.iloc[:, 2:31], df.iloc[:, 1]           #DataFrame con los valores calculados para cada muestra y Series correspondiente de cada fila en X

y.replace('M', -1, inplace=True)
y.replace('B', 1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)         #len(X_train) = 284, len(y_train) = 284
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)    #len(X_test) = 143, len(y_test) = 143, len(X_val) = 142, len(y_val) = 142



def plot_confusion_matrix(confmat):                         #Función que grafica la matriz de confusión
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues,  alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()

#SUPPORT VECTOR MACHINE

linear_svc = SVC(kernel='linear')                           #Kernel lineal
poly_svc = SVC(kernel='poly', gamma=0.2, degree=2)          #Kernel polinomial
rbf_svc = SVC(kernel='rbf', gamma=0.0001, C=7.0)            #Kernel gaussiano
sig_svc = SVC(kernel='sigmoid', gamma=0.0000001, C=10.0)    #Kernel sigmoide
kernels = [linear_svc, poly_svc, rbf_svc, sig_svc]


def train(estimator, x_train, y_train):                     #Función que entrena el modelo
    estimator.fit(x_train, y_train)


def kFold(estimator, x_val, y_val):                         #Validación KFold
    f = 5
    cv = KFold(len(y_val), f, shuffle=True, random_state=0)
    scores = cross_val_score(estimator, x_val, y_val, cv=cv)
    return scores


def validate_svm(x_val, y_val, x_train, y_train, x_test, y_test):    #Función de validación de SVM
    mayor, nombre, best_K = 0, "", 0
    for k in kernels:
        scores = kFold(k, x_val, y_val)

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
    test(best_K, x_train, y_train, x_test, y_test)


def test(best_clf, x_train, y_train, x_test, y_test):
    train(best_clf, x_train, y_train)
    preds = best_clf.predict(x_test)
    print_results(y_test, preds)


def print_results(y_test, preds):
    print("Matriz de Confusión:")
    M = metrics.confusion_matrix(y_test, preds)
    print(M)
    plot_confusion_matrix(M)
    TP, FN, FP, TN = M[0][0], M[0][1], M[1][0], M[1][1]
    accuracy = (TP + TN) / (TP + FN + FP + TN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    f1 = (2*TP)/(2*TP + FP + FN)
    print("Exactitud =", accuracy)
    print("Sensibilidad =", sensitivity)
    print("Especificidad =", specificity)
    print("Valor-F =", f1)


    #plot_decision_regions(X_train_plot, y_train, classifier=best_K)
    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc='upper left')
    #plt.show()

validate_svm(X_val, y_val, X_train, y_train, X_test, y_test)

print("-------------------------------------------------------------------------------------------------")
#LOGISTIC REGRESSION


def validate_lr(x_val, y_val, x_train, y_train, x_test, y_test):
    mayor, c, p = 0, 0, ""
    for C in enumerate((100, 1, 0.01)):
        clf_l1 = LogisticRegression(C=C, penalty='l1', tol=0.01)       #Valores más pequeños de c incrementan la regularización
        clf_l2 = LogisticRegression(C=C, penalty='l2', tol=0.01)
        train(clf_l1, x_train, y_train)
        train(clf_l2, x_train, y_train)
        score_lr1 = clf_l1.score(x_val, y_val)
        score_lr2 = clf_l2.score(x_val, y_val)

        if score_lr1 > mayor:
            mayor = score_lr1
            c = C
            p = 'l1'
        if score_lr2 > mayor:
            mayor = score_lr2
            c = C
            p = 'l2'

    best_lr = LogisticRegression(C=c, penalty=p, tol=0.01)
    test(best_lr, x_train, y_train, x_test, y_test)

validate_lr(X_val, y_val, X_train, y_train, X_test, y_test)

print("-------------------------------------------------------------------------------------------------")

#MULTILAYER PERCEPTRON


def validate_mlp(x_train, y_train, x_test, y_test, x_val, y_val):
    solver = ['lbfgs', 'sgd']
    a = ['tanh', 'logistic', 'relu']
    hl = [(250, 5), (200, 2), (150, 2), (100, 4)]
    mayor = 0
    for s in solver:
        for act in a:
            for h in hl:
                clf = MLPClassifier(solver=s, alpha=1e-5, hidden_layer_sizes=h, activation=act, random_state=1)
                scores_mlp = kFold(clf, x_val, y_val)

                prom = scores_mlp.mean()

                if prom > mayor:
                    mayor = prom
                    S = s
                    A = act
                    HL = h

            print("Mejores parámetros: \n Optimizador=", S, "\n Función de Activación = ", A, "\n Tamaño de la capa oculta = ", HL)
            clf = MLPClassifier(solver=S, alpha=1e-5, hidden_layer_sizes=HL, activation=A, random_state=1)
            test(clf, x_train, y_train, x_test, y_test)

validate_mlp(X_train, y_train, X_test, y_test, X_val, y_val)
