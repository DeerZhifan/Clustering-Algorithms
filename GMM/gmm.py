import numpy as np
import scipy.stats
import matplotlib.pyplot as plt



############################################################
# 产生样本矩阵x，shape = (100, 2)
############################################################
def gen_sample():
    cov1 = [[0.3, 0], [0, 0.1]]
    cov2 = [[0.2, 0], [0, 0.3]]

    mu1 = [0, 1]
    mu2 = [2, 1]

    sample = np.zeros((100, 2))
    sample[:30, :] = np.random.multivariate_normal(mean=mu1, cov=cov1, size=30)
    sample[30:, :] = np.random.multivariate_normal(mean=mu2, cov=cov2, size=70)
    np.savetxt('sample.data', sample)

    plt.plot(sample[:30, 0], sample[:30, 1], 'rs')
    plt.plot(sample[30:, 0], sample[30:, 1], 'bo')
    plt.title("Dataset")
    plt.show()


############################################################
# 数据预处理
# X为样本矩阵
# 将数据进行极差归一化处理
############################################################

def scale_data(x):
    for i in range(x.shape[1]):
        max_    = x[:, i].max()
        min_    = x[:, i].min()
        x[:, i] = (x[:, i] - min_) / (max_ - min_)
    return x



############################################################
# 初始化模型参数
# shape为样本矩阵x的维数（样本数，特征数）
# k为模型的个数
# mu, cov, alpha分别为模型的均值、协方差以及混合系数
############################################################

def init_params(shape, k):
    n, d  = shape
    mu    = np.random.rand(k, d)
    cov   = np.array([np.eye(d)] * k)
    alpha = np.array([1.0 / k] * k)
    return mu, cov, alpha



############################################################
# 第i个模型的高斯密度分布函数
# x 为样本矩阵，行数等于样本数，列数等于特征数
# mu_i, cov_i分别为第i个模型的均值、协方差参数
# 返回样本在该模型下的概率密度值
############################################################

def phi(x, mu_i, cov_i):
    norm = scipy.stats.multivariate_normal(mean=mu_i, cov=cov_i)
    return norm.pdf(x)



############################################################
# E步：计算每个模型对样本的响应度
# x 为样本矩阵，行数等于样本数，列数等于特征数
# mu为均值矩阵， cov为协方差矩阵
# alpha为各模型混合系数组成的一维矩阵
############################################################

def expectation(x, mu, cov, alpha):
    # 样本数，模型数
    n, k = x.shape[0], alpha.shape[0]

    # 计算各模型下所有样本出现的概率矩阵prob，行对应第i个样本，列对应第K个模型
    prob = np.zeros((n, k))
    for i in range(k):
        prob[:, i] = phi(x, mu[i], cov[i])
    prob = np.mat(prob)

    # 计算响应度矩阵gamma，行对应第i个样本，列对应第K个模型
    gamma = np.mat(np.zeros((n, k)))
    for i in range(k):
        gamma[:, i] = alpha[i] * prob[:, i]
    for j in range(n):
        gamma[j, :] /= np.sum(gamma[j, :])
    return gamma



############################################################
# M步：迭代模型参数
############################################################

def maximization(x, gamma):
    # 样本数，特征数
    n, d = x.shape
    # 模型数
    k = gamma.shape[1]

    # 初始化模型参数
    mu = np.zeros((k, d))
    cov = []
    alpha = np.zeros(k)

    # 更新每个模型的参数
    for i in range(k):
        # 第K个模型对所有样本的响应度之和
        ni       = np.sum(gamma[:, i])
        # 更新mu
        mu[i, :] = np.sum(np.multiply(x, gamma[:, i]), axis=0) / ni
        # 更新cov
        cov_i    = (x - mu[i]).T * np.multiply((x - mu[i]), gamma[:, i]) / ni
        cov.append(cov_i)
        # 更新alpha
        alpha[i] = ni / n
    cov = np.array(cov)
    return mu, cov, alpha



############################################################
# 展示聚类结果
############################################################

def show_cluster(cluster):
    color = ['rs', 'bo']

    for i in range(len(cluster)):
        plt.plot(cluster[i][:, 0], cluster[i][:, 1], color[i])
    plt.title("GMM Clustering")
    plt.show()


############################################################
# 高斯混合模型EM算法
# x为样本矩阵，k为模型个数，times为模型迭代次数
############################################################

def gmm_em(k, times):
    # 载入数据集sample.csv
    dataset = np.loadtxt('sample.data')
    x = np.matrix(dataset, copy=True)
    # 数据归一化处理
    x = scale_data(x)
    # 初始化模型参数
    mu, cov, alpha = init_params(x.shape, k)

    # 迭代模型参数
    for i in range(times):
        gamma          = expectation(x, mu, cov, alpha)
        mu, cov, alpha = maximization(x, gamma)

    # 求出当前模型参数下样本的响应矩阵
    gamma = expectation(x, mu, cov, alpha)
    # 样本矩阵中每一行最大值的列索引即为该样本的对应类别
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    # 将每个样本放入对应的类别中
    cluster = []
    for i in range(k):
        item = np.array([dataset[j] for j in range(x.shape[0]) if category[j] == i])
        cluster.append(item)
    show_cluster(cluster)




############################################################
# 程序执行入口
############################################################

if __name__ == "__main__":
    gen_sample()
    k, times = 2, 100
    gmm_em(k, times)