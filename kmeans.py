"""
    Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1. Для этого
    можно воспользоваться функцией img_as_float из модуля skimage. Обратите внимание на этот шаг, так как при работе с
    исходным изображением вы получите некорректный результат.
    Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности в
    пространстве RGB.
    Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241. После выделения кластеров все пиксели,
    отнесенные в один кластер, попробуйте заполнить двумя способами: медианным и средним цветом по кластеру.
    Измерьте качество получившейся сегментации с помощью метрики PSNR. Эту метрику нужно реализовать самостоятельно
     (см. определение).
    Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть не более 20
    кластеров, но не забудьте рассмотреть оба способа заполнения пикселей одного кластера).
    Это число и будет ответом в данной задаче.
"""

#print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

from skimage.io import imread
from skimage import img_as_float


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def ClusterIndicesComp(clustNum, labels_array): #list comprehension
    return np.array([i for i, x in enumerate(labels_array) if x == clustNum])

def recreate_image_mean_colors(km, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = km.cluster_centers_.shape[1]
    codebook = []
    n_clusters = km.n_clusters
    for n_cl in range(n_clusters):
        #n_cl_mean = np.mean([ClusterIndicesComp(n_cl, km.labels_)])
        n_cluster_labels = ClusterIndicesComp(n_cl, km.labels_)
        X = km.labels_[n_cluster_labels]
        max_val =
        n_cl_mean = np.mean(X)
        codebook.append(n_cl_mean)

    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

origin_image = imread('parrots.jpg')
converted_img = img_as_float(origin_image)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
#china = np.array(china, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(converted_img.shape)
assert d == 3
image_array = np.reshape(converted_img, (w * h, d))



for n_clusters in range(3):
    kmeans = KMeans(n_clusters=n_clusters+1, random_state=241, init='k-means++').fit(image_array)

    # Get labels for all points

    labels = kmeans.predict(image_array)

    plt.figure(n_clusters+1)
    plt.clf()
    plt.axis('off')
    plt.title('Clusters')
    plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

    plt.figure(2*(n_clusters+1))
    plt.clf()
    plt.axis('off')
    plt.title('Clusters with mean color')
    plt.imshow(recreate_image_mean_colors(kmeans, labels, w, h))

#codebook_random = shuffle(image_array, random_state=0)[:n_colors]
#print("Predicting color indices on the full image (random)")
#t0 = time()
#labels_random = pairwise_distances_argmin(codebook_random,
#                                          image_array,
#                                          axis=0)
#print("done in %0.3fs." % (time() - t0))



# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(converted_img)
"""
plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
"""