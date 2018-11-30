"""В этом задании вам нужно проследить за изменением качества случайного леса в зависимости от количества деревьев в нем.

Загрузите данные из файла abalone.csv. Это датасет, в котором требуется предсказать возраст ракушки (число колец) по
 физическим измерениям.
Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1. Если вы используете Pandas,
 то подойдет следующий код: data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
Разделите содержимое файлов на признаки и целевую переменную. В последнем столбце записана целевая переменная,
в остальных — признаки.
Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50 (не забудьте
 выставить "random_state=1" в конструкторе). Для каждого из вариантов оцените качество работы полученного леса на
 кросс-валидации по 5 блокам. Используйте параметры "random_state=1" и "shuffle=True" при создании генератора
 кросс-валидации sklearn.cross_validation.KFold. В качестве меры качества воспользуйтесь коэффициентом детерминации
 (sklearn.metrics.r2_score).
Определите, при каком минимальном количестве деревьев случайный лес показывает качество на кросс-валидации выше 0.52.
Это количество и будет ответом на задание.
Обратите внимание на изменение качества по мере роста числа деревьев. Ухудшается ли оно?"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score


#Function cross validates RandomForestRegressor with given number of trees. Return r2_score metric value

def cross_val_score_RF (n_trees, x_data, y_answers):
    clf = RandomForestRegressor(n_estimators=n_trees, random_state=1)
    clf.fit(x_data, y_answers) #Why it is necessary!!!!!!!
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    return cross_val_score(clf, x_data, y_answers, cv=cv, scoring='r2')

data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

print(data.columns)
#the last column index
last_col = len(data.columns) - 1
print(data.columns[last_col])

#Move answers column away from parameters
Y_answers = data.iloc[:,last_col]
data = data.drop(data.columns[last_col], axis = 1)
print(data.columns)

for n_tree in range(50):
    r2 = cross_val_score_RF(n_tree+1, data, Y_answers)
    print('Trees: ', n_tree+1, '\n', 'cross_val_score: \n', r2.mean())
    clf = RandomForestRegressor(n_estimators=n_tree+1, random_state=1)
    clf.fit(data, Y_answers)
    predicted = clf.predict(data)
    print('R2_score: ', r2_score(Y_answers, predicted))
#    print('Trees: %d  r2_score: %0.2f' % (n_tree+1, r2.mean()))
