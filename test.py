import numpy as np
import heapq

a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
ind = np.argpartition(a, -5)[-5:]
print(ind)

top3= heapq.nlargest(3, range(a.size), a.take)
print(a.size)
print (top3)

