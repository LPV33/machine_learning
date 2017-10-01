"""
Редактор Spyder

Это временный скриптовый файл.
"""

import statistics
import numpy as np
from sklearn.linear_model import Perceptron
import sklearn.metrics
import os
print(os.listdir(os.getcwd()))

import pandas

"""Read train data"""
train_data = pandas.read_csv('perceptron-train.csv', names=['Class', 'Sign1', 'Sign2'])

train_class = train_data['Class']
train_signs = train_data.drop(['Class'],inplace=False,axis=1)

"""Read test data"""
test_data = pandas.read_csv('perceptron-test.csv', names=['Class', 'Sign1', 'Sign2'])

test_class = test_data['Class']
test_signs = test_data.drop(['Class'],inplace=False,axis=1)


"Train the Perceptron"

clf = Perceptron(random_state=241)
clf.fit(train_signs, train_class)

"""Check an accuracy of prediction for non-normalized data"""

predictions = clf.predict(test_signs)
accuracy = sklearn.metrics.accuracy_score(test_class, predictions )

print('Non-normalized data ', accuracy, '\n')

"""Check an accuracy of prediction for normalized data"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

"""Normalize tain and test data"""
train_signs_scaled = scaler.fit_transform(train_signs)
test_signs_scaled = scaler.transform(test_signs)

clf1 = Perceptron(random_state=241)
clf1.fit(train_signs_scaled, train_class)

predictions = clf1.predict(test_signs_scaled)
accuracy = sklearn.metrics.accuracy_score(test_class, predictions )

print('Normalized data ', accuracy, '\n')
