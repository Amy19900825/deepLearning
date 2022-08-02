#单隐藏层的平面分类问题
# 在Planar data数据集中分布着众多散点，X保存着每个散点的坐标，Y保存着每个散点对应的标签（0：红色，1：蓝色），
# 要求用单隐层神经网络对该平面图形进行分类。


import numpy as np # 科学计算包
import matplotlib.pyplot as plt # 绘图包
import sklearn  #提供数据挖掘和分析的工具
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets # 加载数据集和一些必要的工具


# 加载数据集
X,Y = load_planar_dataset()
# 绘制散点图，X是含坐标的矩阵，Y是标签 红0蓝1
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.title("scatter pic")
plt.show()
#打印矩阵维度和数量
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]
print("X的维度： " + str(shape_X))
print("Y的维度： " + str(shape_Y))
print("样本数量： " + str(m))

# # 更改数据集
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
#
# datasets = {"noisy_circles": noisy_circles,
#             "noisy_moons": noisy_moons,
#             "blobs": blobs,
#             "gaussian_quantiles": gaussian_quantiles}
#
# dataset = "noisy_moons"
#
# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])
#
# if dataset == "blobs":
#     Y = Y % 2
#
# # plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
#
# #上一语句如出现问题请使用下面的语句：
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
#
# plt.show()
# 逻辑回归验证准确度
# clf = sklearn.linear_model.LogisticRegressionCV() # LogisticRegressionCV使用交叉验证选择正则化系数C，而LogisticRegression每次指定正则化系数
# clf.fit(X.T, Y.T)

# plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y)) #绘制决策边界
# plt.title("Logistic Regression")
# LR_predictions  = clf.predict(X.T)
# print ("逻辑回归的准确度： %d" % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) + "% " + "(正确标记的数据点所占的百分比)")
# print("同为1的数量： " , np.dot(Y, LR_predictions))
# print("同为0的数量： " , np.dot(1 - Y,1 - LR_predictions))

# 定义神经网络结构
def layer_size(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = 4
    return (n_x, n_h, n_y)


# 参数的随机初始化
def initialization(n_x, n_h, n_y):
    # np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.random.randn(n_h, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.random.randn(n_y, 1)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# 前向传播计算结果
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # cache是网络块中的计算结果缓存
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return (A2, cache)


# 计算成本函数
def cost_function(A2, Y, parameters):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    return cost


# 反向传播计算梯度
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))  # 这一步计算式比较难理解
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    # 梯度
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


# 更新参数
def update_para(parameters, grads, learning_rate):
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]
    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    # 更新后的参数
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# 整合函数到神经网络模型中
def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]
    parameters = initialization(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)  # 前向传播计算结果和缓存
        cost = cost_function(A2, Y, parameters)  # 计算成本
        grads = backward_propagation(parameters, cache, X, Y)  # 反向传播计算梯度
        parameters = update_para(parameters, grads, learning_rate=0.5)  # 梯度下降更新参数
        # 如果打印结果，则1000步打印一次结果
        if (print_cost):
            if (i % 1000 == 0):
                print("iter " + str(i) + "cost: " + str(cost))
    return parameters


# 通过前向传播预测结果
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2) # 取整数的好处是不会出现小数标签，绘制图形更好看
    return predictions

# 运行神经网络
n_h = 5
parameters = nn_model(X, Y, n_h, num_iterations=25000, print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y)) # 绘制边界图像
plt.title("Decision Boundary for hidden layer size " + str(n_h))
plt.show()

prediction = predict(parameters, X)
# 准确率的计算仍然是 匹配的点/总点数，匹配点的数量分同0和同1计算，如下
print ('准确率: %d' % float((np.dot(Y, prediction.T) + np.dot(1 - Y, 1 - prediction.T)) / float(Y.size) * 100) + '%')
