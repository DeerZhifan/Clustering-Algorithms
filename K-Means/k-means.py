import numpy as np
import random
import matplotlib.pyplot as plt



############################################################
# 加载数据集dataSet.csv，shape = (80, 2)
############################################################

def load_dataset():
    dataset = np.loadtxt("dataSet.csv")
    return dataset



############################################################
# 展示聚类前数据集分布
############################################################

def show_dataset():
    dataset = load_dataset()
    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    ax.scatter(dataset[:, 0], dataset[:, 1])
    plt.title("Dataset")
    plt.show()



############################################################
# 计算两点之间的欧氏距离并返回
############################################################

def elu_distance(a, b):
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist



############################################################
# 从数据集dataset中随机选取k个数据作为中心(centroids)并返回
############################################################

def initial_centroids(dataset, k):
    dataset   = list(dataset)
    centroids = random.sample(dataset, k)
    return centroids



############################################################
# 对dataset中的每个点item, 计算item与centroids中k个中心的距离
# 根据最小距离将item加入相应的簇中并返回簇类结果cluster
############################################################

def min_distance(dataset, centroids):
    cluster = dict()
    k       = len(centroids)
    for item in dataset:
        a        = item
        flag     = -1
        min_dist = float("inf")
        for i in range(k):
            b    = centroids[i]
            dist = elu_distance(a, b)
            if dist < min_dist:
                min_dist = dist
                flag     = i
        if flag not in cluster.keys():
            cluster[flag] = []
        cluster[flag].append(item)
    return cluster



############################################################
# 根据簇类结果cluster重新计算每个簇的中心
# 返回新的中心centroids
############################################################

def reassign_centroids(cluster):
    # 重新计算k个质心
    centroids = []
    for key in cluster.keys():
        centroid = np.mean(cluster[key], axis=0)
        centroids.append(centroid)
    return centroids



############################################################
# 计算簇内样本与各自中心的距离，累计求和
# sum_dist刻画簇内样本相似度, sum_dist越小则簇内样本相似度越高
############################################################

def closeness(centroids, cluster):
    # 计算均方误差，该均方误差刻画了簇内样本相似度
    # 将簇类中各个点与质心的距离累计求和
    sum_dist = 0.0
    for key in cluster.keys():
        a    = centroids[key]
        dist = 0.0
        for item in cluster[key]:
            b     = item
            dist += elu_distance(a, b)
        sum_dist += dist
    return sum_dist



############################################################
# 展示聚类结果
############################################################

def show_cluster(centroids, cluster):
    # 展示聚类结果
    cluster_color  = ['or', 'ob', 'og', 'ok', 'oy', 'ow']
    centroid_color = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in cluster.keys():
        plt.plot(centroids[key][0], centroids[key][1], centroid_color[key], markersize=12)
        for item in cluster[key]:
            plt.plot(item[0], item[1], cluster_color[key])
    plt.title("K-Means Clustering")
    plt.show()



############################################################
# K-Means算法
# 1、加载数据集
# 2、随机选取k个样本作为中心
# 3、根据样本与k个中心的最小距离进行聚类
# 4、计算簇内样本相似度，并与上一轮相似度进行比较，两者误差小于阈值，则
#    停止运行，反之则更新各类中心，重复步骤3
############################################################

def k_means(k):
    dataset      = load_dataset()
    centroids    = initial_centroids(dataset, k)
    cluster      = min_distance(dataset, centroids)
    current_dist = closeness(centroids, cluster)
    old_dist     = 0

    while abs(current_dist - old_dist) >= 0.00001:
        centroids     = reassign_centroids(cluster)
        cluster       = min_distance(dataset, centroids)
        old_dist      = current_dist
        current_dist  = closeness(centroids, cluster)
    return centroids, cluster


############################################################
# 程序执行入口
############################################################

if __name__ == "__main__":
    show_dataset()
    k = 4
    centroids, cluster = k_means(k)
    show_cluster(centroids, cluster)


