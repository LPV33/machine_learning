# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
import statistics
import numpy as np
from collections import Counter
"""
X = np.random.normal(loc=1, scale=10, size=(1000, 50))
print (X)

m = np.mean(X)
std = np.std(X, axis=0)
X_norm = ((X - m)  / std)
print (m)
print ("--------------------------------------")
print (std)
print ("--------------------------------------")
print (X_norm)


Z = np.array([[4, 5, 0], 
             [1, 9, 3],              
             [5, 1, 1],
             [3, 3, 3], 
             [9, 9, 9], 
             [4, 7, 1]])

r = np.sum(Z, axis=1)
print (r)
print ("--------------------------------------")
print (np.nonzero(r > 10))

 
A = np.eye(3)
B = np.eye(3)
print (A)
print (B)

AB = np.vstack((A, B))
print (AB)
"""

import pandas
passengers = pandas.read_csv('train.csv', index_col='PassengerId')
Male = passengers[passengers['Sex']=='male']
Female = passengers[passengers['Sex']=='female']
Survived = passengers[passengers['Survived']==1]
FirstClass = passengers[passengers['Pclass']==1]
pearson = passengers.drop(['Survived', 'Pclass','Name','Sex','Age','Ticket','Fare','Cabin','Embarked'],inplace=False,axis=1)

print(pearson.corr())
print('Men ', len(Male))
print('Women ', len(Female))
print ('Survived ratio ',round(sum(passengers['Survived'])/len(passengers), 4))
print ('1st class passengers ratio ',round(len(FirstClass)/len(passengers), 4))
print ('Average age ',round(np.mean(passengers['Age'].dropna()), 4))
print ('Median age ',round(statistics.median(passengers['Age'].dropna()),4))



print ('********************************************')
"""
for line in Female.Name:
    print (line)
"""

Names =[]

for line in Female.Name:

    if 'Mrs' in line:
        marriage = 'Missis'
        name = line.split('(')
        if len(name) > 1:
            firstname = name[1].split()[0]
        else:
            firstname = name[0]
    elif 'Miss'in line:
        marriage = 'Miss'
        name = line.split('.')
        firstname = name[1].split()[0]
    else:
        continue
    firstname = firstname.replace('(','')
    firstname = firstname.replace(')','')
    
    if len(firstname.split()) > 1:
        continue
    
    Names.append(firstname) 
    

c = Counter(Names)
print ('Most common female names ',c.most_common(3))

cand = passengers.drop(['Name','Ticket','Cabin','Embarked', 'SibSp', 'Parch'],inplace=False,axis=1) 
cand.dropna(inplace=True)
target_func=cand['Survived']


Sex = []
for line in cand.Sex:
    if 'female' in line:
        sex_ = 1
    else:
        sex_ = 0
        
    Sex.append(sex_)


cand.drop(['Survived','Sex'], inplace=True,axis=1)
cand['Sex'] = Sex
    
print (cand)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=241)
clf.fit(cand, target_func)
importances = clf.feature_importances_

print (importances)

