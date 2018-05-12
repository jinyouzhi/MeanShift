import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import MeanShift


def load_data(path, label_num=3, feature_num=2):
    f = open(path)
    data = []
    label = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        label_tmp = []
        if len(lines) != label_num + feature_num:
            continue
        for i in range(label_num):
            label_tmp.append(lines[i])
        for i in range(feature_num):
            data_tmp.append(float(lines[label_num + i]))
        label.append(label_tmp)
        data.append(data_tmp)
    f.close()
    return label, data


def show(perference, data, cluster_centers, n_clusters_):
    # #############################################################################
    # Plot result
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = cluster == k
        cluster_center = cluster_centers[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title(perference + '     Estimated number of clusters: %d(cluster)/%d(total)' % (n_clusters_,len(data)))
    plt.savefig("image/" + perference.strip().replace('/', '') + ".jpg")
    #plt.show()


if __name__ == "__main__":
    # 导入数据集
    path = "./data.txt"
    label, data = load_data(path, 3, 2)

    fout = open('res.txt', 'w')
    start = epoch = 0
    ms = MeanShift(bandwidth=0.03, n_jobs=-1)
    for i in range(len(label)):
        if i % 100 == 0:
            print(i)
        if i + 1 == len(label) or label[i][1] != label[i + 1][1]:
            epoch = epoch + 1
            cur = data[start:i + 1]
            ms.fit(cur)
            cluster = ms.labels_
            cluster_centers = ms.cluster_centers_
            n_clusters_ = len(np.unique(cluster))
            print("epoch:" + str(epoch) + '\tTotal:' + str(len(cur)) + '\tcluster:' + str(n_clusters_))
            for j in range(len(cur)):
                fout.write(label[j+start][0] + '\t' + label[j+start][1] + '\t' + label[j+start][2] + '\t' + str(cur[j][0]) + '\t' + str(
                    cur[j][1]) + '\t' + str(cluster[j]) + '\n')

            start = i + 1
            show(label[i][2], np.mat(cur), cluster_centers, n_clusters_)
    fout.close()
