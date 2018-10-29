"""агрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка pred).
Заполните таблицу ошибок классификации:
Actual Positive	Actual Negative
Predicted Positive	TP	FP
Predicted Negative	FN	TN
Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям. Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.

3. Посчитайте основные метрики качества классификатора:

Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
Precision (точность) — sklearn.metrics.precision_score
Recall (полнота) — sklearn.metrics.recall_score
F-мера — sklearn.metrics.f1_score
В качестве ответа укажите эти четыре числа через пробел.

4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:

для логистической регрессии — вероятность положительного класса (колонка score_logreg),
для SVM — отступ от разделяющей поверхности (колонка score_svm),
для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
для решающего дерева — доля положительных объектов в листе (колонка score_tree).
Загрузите этот файл.

5. Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.

6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?"""

import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_fscore_support as score
"""
import os
print(os.listdir(os.getcwd()))
"""
import pandas
"""Read data data"""
data = pandas.read_csv('classification.csv', names=['true', 'pred'])

data = data.drop([0], axis = 0)
print ('Data\n', data)

TP = 0
TN = 0
FP = 0
FN = 0
for index, row in data.iterrows():
    if (row['true'] == '1' ):
        if (row['pred'] == '1'):
            TP = TP + 1
            continue
        FN = FN + 1
        continue
    if (row['pred'] == '1'):
        FP = FP + 1
        continue
    TN = TN + 1

print ('TP = ', TP, 'TN = ', TN, 'FP = ', FP, 'FN = ', FN, '\n' )

#Accuracy

print ('Accuracy ', accuracy_score(data['true'], data['pred']), 'To check ', (TP+TN)/(TP+FP+FN+TN) )
Precision = TP/(TP+FP)
print ('Precision ', precision_score(data['true'], data['pred'], average=None), 'To check ', Precision )
Recall = TP/(TP+FN)
print ('Recall ', recall_score(data['true'], data['pred'], average=None), 'To check ', Recall )
print ('F-measure ', f1_score(data['true'], data['pred'], average=None), 'To check ', 2*Precision*Recall/(Precision+Recall) )

precision, recall, fscore, support = score(data['true'], data['pred'])

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
