# coding: utf-8
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def read_data():
    points = []
    with open('testSet.txt', encoding='utf-8') as fp:
        for line in fp:
            x, y = line.strip().split()
            points.append([float(x), float(y)])
    return np.array(points)


def train(data, n_clusters=3):
    model = KMeans(n_clusters)
    model.fit(data)
    pred = model.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=pred, s=50, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    data = read_data()
    train(data, 4)
