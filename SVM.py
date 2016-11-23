import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('wdbc.data.txt', header=None)   #Dataframe: conjunto de datos

X, y = df.iloc[:, 2:31], df.iloc[:, 1]           #DataFrame con los valores calculados para cada muestra y Series correspondiente de cada fila en X

y.replace('M', 1, inplace=True)
y.replace('B', 0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)         #len(X_train) = 284, len(y_train) = 284
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)    #len(X_test) = 143, len(y_test) = 143, len(X_val) = 142, len(y_val) = 142

best_results = []

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
print("MÁQUINA DE VECTORES DE SOPORTE")

linear_svc = SVC(kernel='linear')                           #Kernel lineal
poly_svc = SVC(kernel='poly', gamma=0.2, degree=2)          #Kernel polinomial
rbf_svc = SVC(kernel='rbf', gamma=0.0001, C=7.0)            #Kernel gaussiano
sig_svc = SVC(kernel='sigmoid', gamma=0.0000001, C=10.0)    #Kernel sigmoide
kernels = [linear_svc, poly_svc, rbf_svc, sig_svc]


def train(estimator, x_train, y_train):                     #Función que entrena el modelo
    estimator.fit(x_train, y_train)


def kFold(estimator, x_val, y_val):                         #Validación KFold
    f = 5
    cv = KFold(shuffle=True, n_splits=f, random_state=0)
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
    M = metrics.confusion_matrix(y_test, preds)
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
    best_results.append(accuracy)


    #plot_decision_regions(X_train_plot, y_train, classifier=best_K)
    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc='upper left')
    #plt.show()

validate_svm(X_val, y_val, X_train, y_train, X_test, y_test)

print("-------------------------------------------------------------------------------------------------")
print("REGRESIÓN LOGÍSTICA REGULARIZADA")
#LOGISTIC REGRESSION


def validate_lr(x_val, y_val, x_train, y_train, x_test, y_test):
    mayor, vc, p, c = 0, [100, 1, 0.01], "", 0
    for C in vc:
        clf_l1 = LogisticRegression(C=C, penalty='l1', tol=0.01)       #Valores más pequeños de c incrementan la regularización
        clf_l2 = LogisticRegression(C=C, penalty='l2', tol=0.01)
        train(clf_l1, x_train, y_train)
        train(clf_l2, x_train, y_train)
        score_lr1 = kFold(clf_l1, x_val, y_val)
        score_lr2 = kFold(clf_l2, x_val, y_val)
        score_lr1 = score_lr1.mean()
        score_lr2 = score_lr2.mean()

        if score_lr1 > mayor:
            mayor = score_lr1
            c = C
            p = 'l1'
        if score_lr2 > mayor:
            mayor = score_lr2
            c = C
            p = 'l2'

    best_lr = LogisticRegression(C=c, penalty=p, tol=0.01)
    print("Mejores parámetros: \nRegularización =", p, "\nC =", c)
    test(best_lr, x_train, y_train, x_test, y_test)

validate_lr(X_val, y_val, X_train, y_train, X_test, y_test)

print("-------------------------------------------------------------------------------------------------")
print("PERCEPTRON MULTICAPA")
#MULTILAYER PERCEPTRON


def validate_mlp(x_train, y_train, x_test, y_test, x_val, y_val):
    solver = ['lbfgs', 'sgd', 'adam']
    a = ['tanh', 'logistic', 'relu']
    hl = [(250, 5), (200, 2), (150, 2), (100, 2)]
    mayor, S, A, HL = 0, 0, 0, 0
    for s in solver:
        for act in a:
            for h in hl:
                clf = MLPClassifier(solver=s, alpha=1e-5, hidden_layer_sizes=h, activation=act, random_state=1,
                                    max_iter=2200)
                scores_mlp = kFold(clf, x_val, y_val)

                prom = scores_mlp.mean()

                if prom > mayor:
                    mayor = prom
                    S = s
                    A = act
                    HL = h

    print("Mejores parámetros: \nOptimizador =", S, "\nFunción de Activación = ", A, "\nTamaño de la capa oculta = ",
          HL)
    clf = MLPClassifier(solver=S, alpha=1e-5, hidden_layer_sizes=HL, activation=A, random_state=1)
    test(clf, x_train, y_train, x_test, y_test)

validate_mlp(X_train, y_train, X_test, y_test, X_val, y_val)

print("-------------------------------------------------------------------------------------------------")
maxACC = np.argmax(best_results)
if maxACC == 0:
    print("El algoritmo con mayor exactitud es MÁQUINA DE VECTORES DE SOPORTE")
elif maxACC == 1:
    print("El algoritmo con mayor exactitud es REGRESIÓN LOGÍSTICA REGULARIZADA")
else:
    print("El algoritmo con mayor exactitud es PERCEPTRON MULTICAPA")
