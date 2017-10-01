"""
    Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
    (инструкция приведена выше). Обратите внимание, что загрузка данных может занять несколько минут

    Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам
    вычислить TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем множестве используют
    информацию из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения целевой
    переменной из теста. На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны
    на момент обучения, и поэтому можно ими пользоваться при обучении алгоритма.
    Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром
    kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для KFold.
    В качестве меры качества используйте долю верных ответов (accuracy).
    Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
    Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC). Они являются
    ответом на это задание. Укажите эти слова через запятую или пробел, в нижнем регистре, в лексикографическом порядке.
"""

import statistics
import numpy as np
import pandas
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sklearn.metrics
import os
print(os.listdir(os.getcwd()))

"""Read raw data"""
from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )



tf_idf = TfidfVectorizer()

mapped_data = tf_idf.fit_transform(newsgroups.data, y=None)


print (mapped_data)

grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

gs.fit(mapped_data, newsgroups.target)

"""
Looking for the best parameter and min C
"""

best_param = 0
min_C = 100000
for a in gs.grid_scores_:
    print('A.mean ', a.mean_validation_score, "\n")
    print('A.params ', a.parameters['C'], '\n')

    if a.mean_validation_score > best_param:
        best_param = a.mean_validation_score
        min_C = a.parameters['C']
    if a.mean_validation_score == best_param:
        if min_C > a.parameters['C']:
            min_C = a.parameters['C']


print ('best_param = ', best_param,'\n')
print ('min_C = ', min_C,'\n')


clf = SVC(kernel='linear', random_state=241, C=min_C)
clf.fit(mapped_data, newsgroups.target)

feature_mapping = tf_idf.get_feature_names()

weight = clf.coef_[0]

print (weight)
"""d = dict.fromkeys(feature_mapping, weight)

print(d)"""