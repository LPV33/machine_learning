import numpy as np
import heapq

a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
ind = np.argpartition(a, -5)[-5:]
print(ind)

top3= heapq.nlargest(3, range(a.size), a.take)
print(a.size)
print (top3)

# входные данные
X = np.array([[0, 1],
              [0, 1],
              [1, 0],
              [1, 0]])

print (X)

# выходные данные
y = np.array([[0, 0, 1, 1]]).T

print(y)

# сделаем случайные числа детерминированными
np.random.seed(1)
synapse_0 = 2*np.random.random((2,1)) - 1

print(synapse_0)

print(np.dot(X,synapse_0))