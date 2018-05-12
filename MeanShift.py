import math
import sys
import numpy as np

MIN_DISTANCE = 0.000001  # mini error
index = 0


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


def gaussian_kernel(distance, bandwidth):
    m = np.shape(distance)[0]
    right = np.mat(np.zeros((m, 1)))
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))

    gaussian_val = left * right
    return gaussian_val


def shift_point(point, points, kernel_bandwidth):
    points = np.mat(points)
    m, n = np.shape(points)
    # 计算距离
    point_distances = np.mat(np.zeros((m, 1)))
    for i in range(m):
        # point_distances[i, 0] = np.sqrt((point - points[i]) * (point - points[i]).T)
        point_distances[i, 0] = geograph_dist(point, points[i])

    # 计算高斯核
    point_weights = gaussian_kernel(point_distances, kernel_bandwidth)

    # 计算分母
    all = 0.0
    for i in range(m):
        all += point_weights[i, 0]

    # 均值偏移
    point_shifted = point_weights.T * points / all
    return point_shifted


def euclidean_dist(pointA, pointB):
    # 计算pointA和pointB之间的欧式距离
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)


def geograph_dist(pointA, pointB):
    # 计算pointA和pointB之间的地理距离（vincenty法）
    from geopy.distance import vincenty
    return vincenty(pointA.tolist(), pointB.tolist()).meters


def distance_to_group(point, group):
    min_distance = 100000.0
    for pt in group:
        dist = geograph_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance


def group_points(mean_shift_points):
    global index
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(mean_shift_points[i, j]))

        item_1 = "_".join(item)
        print(item_1)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(mean_shift_points[i, j]))

        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])
    return group_assignment


def train_mean_shift(points, kenel_bandwidth=2):
    # shift_points = np.array(points)
    mean_shift_points = np.mat(points)
    max_min_dist = 100
    iter = 0
    m, n = np.shape(mean_shift_points)
    need_shift = [True] * m

    # cal the mean shift vector
    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iter += 1
        print("iter : " + str(iter))
        for i in range(0, m):
            # 判断每一个样本点是否需要计算偏置均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kenel_bandwidth)
            dist = geograph_dist(p_new, p_new_start)

            if dist > max_min_dist:  # record the max in all points
                max_min_dist = dist
            if dist < MIN_DISTANCE:  # no need to move
                need_shift[i] = False

            mean_shift_points[i] = p_new
    # 计算最终的group
    group = group_points(mean_shift_points)

    return np.mat(points), mean_shift_points, group


if __name__ == "__main__":
    # 导入数据集
    path = "./data.txt"
    label, data = load_data(path, 3, 2)

    start = 0
    first = True
    for i in range(len(label)):
        if i % 100 == 0:
            print(i)
        if i + 1 == len(label) or label[i][1] != label[i + 1][1]:
            # 训练，h=2
            epoch = 0
            last = 0
            if first:
                points, shift_points, cluster = train_mean_shift(data[start:i + 1], 1000)
                epoch = epoch + 1
                print("epoch:" + str(epoch) + '\t' + str(index - last))
                last = index
                first = False
            else:
                _points, _shift_points, _cluster = train_mean_shift(data[start:i + 1], 1000)
                epoch = epoch + 1
                print("epoch:" + str(epoch) + '\t' + str(index - last))
                last = index
                points = np.row_stack((points, _points))
                shift_points = np.row_stack((shift_points, _shift_points))
                cluster.extend(_cluster)
            start = i + 1
    fout = open('res.txt', 'w')
    for i in range(len(label)):
        fout.write(label[i][0] + '\t' + label[i][1] + '\t' + label[i][2] + '\t' + str(data[i][0]) + '\t' + str(
            data[i][1]) + '\t' + str(cluster[i]) + '\n')
        print("%5.2f,%5.2f\t%5.2f,%5.2f\t%i" % (
            points[i, 0], points[i, 1], shift_points[i, 0], shift_points[i, 1], cluster[i]))
    fout.close()
