import numpy as np
import random
import matplotlib.pyplot as plt



############################################################
# 西瓜数据集4.0  编号，密度，含糖率
# 数据集来源：《机器学习》第九章 周志华教授
############################################################

data = '''
1,0.697,0.460,
2,0.774,0.376,
3,0.634,0.264,
4,0.608,0.318,
5,0.556,0.215,
6,0.403,0.237,
7,0.481,0.149,
8,0.437,0.211,
9,0.666,0.091,
10,0.243,0.267,
11,0.245,0.057,
12,0.343,0.099,
13,0.639,0.161,
14,0.657,0.198,
15,0.360,0.370,
16,0.593,0.042,
17,0.719,0.103,
18,0.359,0.188,
19,0.339,0.241,
20,0.282,0.257,
21,0.748,0.232,
22,0.714,0.346,
23,0.483,0.312,
24,0.478,0.437,
25,0.525,0.369,
26,0.751,0.489,
27,0.532,0.472,
28,0.473,0.376,
29,0.725,0.445,
30,0.446,0.459'''



############################################################
# 加载数据集
############################################################

def load_dataset(data):
    data_   = data.strip().split(',')
    dataset = [(float(data_[i]), float(data_[i+1])) for i in range(1, len(data_)-1, 3)]
    return dataset



############################################################
# 展示聚类前数据集分布
############################################################

def show_dataset(dataset):
    for item in dataset:
        plt.plot(item[0], item[1], 'ob')
    plt.title("Dataset")
    plt.show()



############################################################
#  计算两个点之间欧氏距离
############################################################

def elu_distance(a, b):
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist



############################################################
# DBSCAN算法：基于密度可达关系导出最大密度相连样本集合
# 1、根据给定的邻域参数（e, min_points）找出所有的核心对象
# 2、以任一核心对象为出发点，找出由其密度可达的样本生成聚类簇
# 3、当所有核心对象均被访问过时停止运行
############################################################

def dbscan(dataset, e, min_points):
    # 聚类个数
    k         = 0
    # 核心对象集合
    omega     = set()
    # 未访问样本集合
    not_visit = set(dataset)
    # 聚类结果
    cluster   = dict()

    # 遍历样本集找出所有核心对象
    for di in dataset:
        if len([dj for dj in dataset if elu_distance(di, dj) <= e]) >= min_points:
            omega.add(di)

    while len(omega):
        # 记录当前未访问样本集合
        not_visit_old = not_visit
        # 随机选取一个核心对象core
        core = list(omega)[random.randint(0, len(omega)-1)]
        not_visit  = not_visit - set(core)
        # 初始化队列，存放核心对象或样本
        core_deque = []
        core_deque.append(core)


        while len(core_deque):
            coreq = core_deque[0]
            # 找出以coreq邻域内的样本点
            coreq_neighborhood = [di for di in dataset if elu_distance(di, coreq) <= e]

            # 若coreq为核心对象，则通过求交集方式将其邻域内且未被访问过的样本找出
            if len(coreq_neighborhood) >= min_points:
                intersection = not_visit & set(coreq_neighborhood)
                core_deque  += list(intersection)
                not_visit    = not_visit - intersection

            core_deque.remove(coreq)
        k += 1
        Ck = not_visit_old - not_visit
        omega = omega - Ck
        cluster[k] = list(Ck)
    return cluster




############################################################
# 展示聚类结果
############################################################

def show_cluster(cluster):
    colors = ['or', 'ob', 'og', 'ok', 'oy', 'ow']
    for key in cluster.keys():
        for item in cluster[key]:
            plt.plot(item[0], item[1], colors[key])
    plt.title("DBSCAN Clustering")
    plt.show()




############################################################
# 程序执行入口
############################################################

if __name__ == "__main__":
    dataset = load_dataset(data)
    show_dataset(dataset)
    e, min_points = 0.11, 5
    cluster = dbscan(dataset, e, min_points)
    print(cluster)
    show_cluster(cluster)
