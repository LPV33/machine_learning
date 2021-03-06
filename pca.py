"""
В этом задании понадобится измерять схожесть двух наборов величин. Если имеется набор пар измерений (например, одна пара —
предсказания двух классификаторов для одного и того же объекта), то охарактеризовать их зависимость друг от друга можно
с помощью корреляции Пирсона. Она принимает значения от -1 до 1 и показывает, насколько данные величины линейно зависимы.
Если корреляция равна -1 или 1, то величины линейно выражаются друг через друга. Если она равна нулю, то линейная
 зависимость между величинами отсутствует.
В этом задании мы будем работать с данными о стоимостях акций 30 крупнейших компаний США. На основе этих данных можно
 оценить состояние экономики, например, с помощью индекса Доу-Джонса. Со временем состав компаний, по которым строится
 индекс, меняется. Для набора данных был взят период с 23.09.2013 по 18.03.2015, в котором набор компаний был
 фиксирован (подробнее почитать о составе можно по ссылке из материалов).
Одним из существенных недостатков индекса Доу-Джонса является способ его вычисления — при подсчёте индекса цены
 входящих в него акций складываются, а потом делятся на поправочный коэффициент. В результате, даже если одна компания
 заметно меньше по капитализации, чем другая, но стоимость одной её акции выше, то она сильнее влияет на индекс.
 Даже большое процентное изменение цены относительно дешёвой акции может быть нивелировано незначительным в процентном
 отношении изменением цены более дорогой акции.
Инструкция по выполнению
Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.
На загруженных данных обучите преобразование PCA с числом компоненты равным 10. Скольких компонент хватит, чтобы объяснить 90% дисперсии?
Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv. Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
"""



import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

X = pd.read_csv('close_prices.csv')
X.drop(['date'], inplace=True,axis=1)

pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)

X1 = pca.transform(X)
print(X1)
#Take the first component
fst_comp = X1[:,0]
print(fst_comp)

Y = pd.read_csv('djia_index.csv')
Y.drop(['date'], inplace=True,axis=1)

print(fst_comp.shape)
print(Y['^DJI'].shape)
print(np.corrcoef(fst_comp, Y['^DJI']))

#contribution of parameters into the components
print (pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9','PC-10']))




