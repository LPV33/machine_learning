"""
Инструкция по выполнению

    Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy (параметр values
     у датафрейма). В первой колонке файла с данными записано, была или нет реакция.
     Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы, такие как размер, форма и т.д.
     Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами test_size = 0.8 и
     random_state = 241.
    Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 и для каждого
     значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
        Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке на каждой
        итерации.
        Преобразуйте полученное предсказание с помощью сигмоидной функции по формуле 1 / (1 + e^{−y_pred}),
        где y_pred — предсказанное значение.
        Вычислите и постройте график значений log-loss (которую можно посчитать с помощью
        функции sklearn.metrics.log_loss) на обучающей и тестовой выборках, а также найдите минимальное значение
        метрики и номер итерации, на которой оно достигается.

3. Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации:
переобучение (overfitting) или недообучение (underfitting)? В ответе укажите одно из слов overfitting либо underfitting.

4. Приведите минимальное значение log-loss на тестовой выборке и номер итерации, на котором оно достигается,
при learning_rate = 0.2.

5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций, на котором
достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и остальными
параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса? (Не забывайте, что
предсказания нужно получать с помощью функции predict_proba. В данном случае брать сигмоиду от оценки вероятности
класса не нужно)
"""


import numpy as np
import pandas as pd
from math import exp

import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


data = pd.read_csv('gbm-data.csv')
data = data.astype(np.float32)

#Move answers column away from parameters
#y = data.iloc[:,0]
#X = data.drop(data.columns[0], axis = 1)

y = data['Activity']
X = data.drop(['Activity'], axis = 1)


print(data.columns)

#Split data to train and test samples
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.8, random_state=241)

#Parameters for GradientBoostingRegressor
original_params = {'n_estimators': 250, 'verbose': True, 'random_state': 241}

plt.figure()

for label, color, setting in [('learning_rate=1', 'orange',
                               {'learning_rate': 1.0}),
                              ('learning_rate=0.5', 'turquoise',
                               {'learning_rate': 0.5}),
                              ('learning_rate==0.3', 'blue',
                               {'learning_rate': 0.3}),
                              ('learning_rate=0.2', 'gray',
                               {'learning_rate': 0.2}),
                              ('learning_rate=0.1', 'magenta',
                               {'learning_rate': 0.1})]:
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    #Loss on test data
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        sigm_y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        test_deviance[i] = metrics.log_loss(y_test, sigm_y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', color=color, label=label)

#For the second part of the task
    if (label == 'learning_rate=0.2'):
        ind_min = np.argmin(test_deviance)
        iter_num = ind_min + 1
        print ('Min loss at learning_rate: ',test_deviance[ind_min], ' Iteration: ', iter_num )

    #Loss on train data
    for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
        sigm_y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        test_deviance[i] = metrics.log_loss(y_train, sigm_y_pred)

    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5],test_deviance[::5], '-',color=color, label=label+' train set')


plt.legend(loc='upper left')
plt.xlabel('Boosting Iterations')
plt.ylabel('Test/Train Set Deviance')

plt.show

#For the third part of task
clf = ensemble.RandomForestClassifier(n_estimators=iter_num, random_state=241)
clf.fit(X_train, y_train)

loss = metrics.log_loss(y_test, clf.predict_proba(X_test))

print ('Random Forest Loss: ',loss)
