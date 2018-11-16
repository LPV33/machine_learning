"""Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv
(либо его заархивированную версию salary-train.zip).
Проведите предобработку:
Приведите тексты к нижнему регистру (text.lower()).
Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text).
Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
Примените TfidfVectorizer для преобразования текстов в векторы признаков. Оставьте только те слова, которые встречаются
хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
Код для этого был приведен выше.
Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и
категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией
 scipy.sparse.hstack.
3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. Целевая переменная записана в столбце SalaryNormalized.

4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. Значения полученных прогнозов являются
ответом на задание. Укажите их через пробел."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('salary-train.csv')
test_data = pd.read_csv('salary-test-mini.csv')

#Make text to lower case
train_data['LocationNormalized'] = train_data['LocationNormalized'].str.lower()
train_data['ContractTime'] = train_data['ContractTime'].str.lower()
train_data['FullDescription'] = train_data['ContractTime'].str.lower()

test_data['LocationNormalized'] = test_data['LocationNormalized'].str.lower()
test_data['ContractTime'] = test_data['ContractTime'].str.lower()
test_data['FullDescription'] = test_data['ContractTime'].str.lower()

#Replace all symbols except letters and digits by spaces
test_data['FullDescription'] = test_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test_data['FullDescription'] = test_data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

print(train_data.columns.values.tolist())

tf_idf = TfidfVectorizer(min_df=5)

train_full_descr_transformed = tf_idf.fit_transform(train_data['FullDescription'].values.astype('U'), y=None)
test_full_descr_transformed = tf_idf.transform(test_data['FullDescription'].values.astype('U'))

train_data['LocationNormalized'].fillna('nan', inplace=True)
train_data['ContractTime'].fillna('nan', inplace=True)

from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()

X_train_categ = enc.fit_transform(train_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

"""
print ('X_train_categ size: ', X_train_categ.toarray().shape[0], '\n')
print ('X_test_categ size: ', X_test_categ.shape[0], '\n')
print ('X_test_categ: ', X_test_categ.toarray(), '\n')
print ('X_train_categ: ', X_train_categ, '\n')
print ('train_full_descr_transformed size: ', train_full_descr_transformed.shape[0], '\n')
print ('train_data[SalaryNormalized]: shape', np.matrix(train_data['SalaryNormalized'].values).T.shape[0], '\n')
"""

Y_answers = np.matrix(train_data['SalaryNormalized'].values).T

from scipy.sparse import hstack
transformed_data = hstack([train_full_descr_transformed, X_train_categ.toarray(), Y_answers])


#print (transformed_data)