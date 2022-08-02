#使用matplotlib.pyplot画图
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.01)
y = x ** 2

plt.plot(x, y)
plt.show()