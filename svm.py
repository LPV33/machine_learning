"""
Инструкция по выполнению

    Загрузите выборку из файла svm-data.csv. В нем записана двумерная выборка (целевая переменная указана
    в первом столбце, признаки — во втором и третьем).
    Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241. Такое значение параметра
    нужно использовать, чтобы убедиться, что SVM работает с выборкой как с линейно разделимой. При более низких
    значениях параметра алгоритм будет настраиваться с учетом слагаемого в функционале, штрафующего за маленькие
    отступы, из-за чего результат может не совпасть с решением классической задачи SVM для линейно азделимой выборки.
    Найдите номера объектов, которые являются опорными (нумерация с единицы). Они будут являться ответом на задание.
    Обратите внимание, что в качестве ответа нужно привести номера объектов в возрастающем порядке через запятую
    или пробел. Нумерация начинается с 1.
"""

import statistics
import numpy as np
from sklearn.svm import SVC
import sklearn.metrics
import os
print(os.listdir(os.getcwd()))

import pandas

"""Read raw data"""
train_data = pandas.read_csv('svm-data.csv', names=['Class', 'Sign1', 'Sign2'])

train_class = train_data['Class']
train_signs = train_data.drop(['Class'],inplace=False,axis=1)

"Train the SVC with linear kernel and C=100000"

clf = SVC(kernel='linear', random_state=241, C=100000)
clf.fit(train_signs, train_class)

print('Support vectors numbers ', clf.support_, '\n')

