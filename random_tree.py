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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


#Function teaching RandomForestRegressor with given number of trees. Return r2_score metric value

def teach_RFR (n_trees, x_data, y_answers, random_state, shuffle):
    clf = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
    w_old = w0
    while i < limit:
        w_new = grad_calc(w_old[0], w_old[1], data, k, C)
        if (max(abs(w_new - w_old)) <= accuracy):
            print ('Accuracy', abs(w_new - w_old), '\n')
            return np.append(w_new, [i])
        i = i + 1
        w_old = w_new
    return np.append(w_old, [i])

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