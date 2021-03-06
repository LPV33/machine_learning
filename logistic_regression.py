"""
Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой принимает значения -1 или 1.
Убедитесь, что выше выписаны правильные формулы для градиентного спуска. Обратите внимание, что мы используем полноценный градиентный спуск, а не его стохастический вариант!
Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10) логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов на соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций десятью тысячами.
Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании? Эти величины будут ответом на задание. В качестве ответа приведите два числа через пробел. Обратите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей, подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией: a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).
Попробуйте поменять длину шага. Будет ли сходиться алгоритм, если делать более длинные шаги? Как меняется число итераций при уменьшении длины шага?
Попробуйте менять начальное приближение. Влияет ли оно на что-нибудь?
Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой, например, 0.421. При необходимости округляйте дробную часть до трех знаков.

Ответ на каждое задание — текстовый файл, содержащий ответ в первой строчке. Обратите внимание, что отправляемые файлы не должны содержать перевод строки в конце. Данный нюанс является ограничением платформы Coursera. Мы работаем над тем, чтобы убрать это ограничение.
"""

import math
import numpy as np
from sklearn.metrics import roc_auc_score
"""
import os
print(os.listdir(os.getcwd()))
"""
import pandas
"""Read data data"""
data = pandas.read_csv('data_logistic.csv', names=['Y', 'X1', 'X2'])

def grad_calc (w1_prev, w2_prev, data, k, C):
    l = len(data)
    w1 = w1_prev
    w2 = w2_prev

    for index, row in data.iterrows():
        y = row['Y']
        x1 = row['X1']
        x2 = row['X2']
        w1 = w1 + k / l * y * x1 * (1 - 1 / (1 + math.exp((-1) * y * (w1_prev * x1 + w2_prev * x2))))
        w2 = w2 + k / l * y * x2 * (1 - 1 / (1 + math.exp((-1) * y * (w1_prev * x1 + w2_prev * x2))))

    w1 = w1 - k * C * w1_prev
    w2 = w2 - k * C * w2_prev
    return np.array([w1, w2])


def grad_iterations (k, C, accuracy, limit, w0, data):
    i = 0
    w_old = w0
    while i < limit:
        w_new = grad_calc(w_old[0], w_old[1], data, k, C)
        if (max(abs(w_new - w_old)) <= accuracy):
            print ('Accuracy', abs(w_new - w_old), '\n')
            return np.append(w_new, [i])
        i = i + 1
        w_old = w_new
    return np.append(w_old, [i])

def predicted_probability(w1, w2, data):
    a = 1/(1 + np.exp(- w1 * data['X1'] - w2 * data['X2']))
    return a


k = 0.15
accuracy = 0.00001
iterations = 10000
C = 10

w0 = np.array([0, 0])

weights = grad_iterations(k, C, accuracy, iterations, w0, data )

print ('C = ', C,' Weights: ', weights, '\n')

print ('Accuracy:\n', roc_auc_score(data['Y'], predicted_probability(weights[0], weights[1], data)))

C = 0

weights = grad_iterations(k, C, accuracy, iterations, w0, data )

print ('C = ', C,' Weights: ', weights, '\n')

print ('Accuracy:\n', roc_auc_score(data['Y'], predicted_probability(weights[0], weights[1], data)))


