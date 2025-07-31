import matplotlib.pyplot as plt
import numpy as np
from numpy import mean

data = [(2.5,2.4), (0.5,0.7), (2.2,2.9), (1.9,2.2), (3.1,3.0), (2.3, 2.7), (2, 1.6), (1, 1.1), (1.5, 1.6), (1.1, 0.9)]
x = [i[0] for i in data]
y = [i[1] for i in data]

# print(x)
# print(y)
# plt.scatter(x,y)
# plt.title('raw data')
# plt.show()

x1 = x - mean(x)  # 中心化
# print(x1)
# print(mean(x))
x2 = x1 / np.linalg.norm(x1)  # 0,1标准化
# print(x1)

y1 = y - mean(y)  # 中心化
y2 = y1 / np.linalg.norm(y1)  # 0,1标准化
